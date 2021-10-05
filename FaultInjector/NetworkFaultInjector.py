import pickle

import numpy as np


class EmptyFaultList(Exception):
    pass


class OutOfFaultList(Exception):
    pass


class NetworkFaultInjector:

    def __init__(self, network, seed):
        """
            Initialize the fault injector
            :param network: the network to be injected
            :param seed: the seed of the run
        """
        self.network = network
        self.seed = seed

        self.rng = np.random.default_rng(seed=self.seed)  # The dedicated random number generator

        self.layer_shape = [(layer.get_weights()[0].shape if (len(layer.get_weights()) > 0) else ())
                            for layer in self.network.layers]  # List of the shape of all layers

        self.fault_list = []  # The list containing all the fault generated for this run
        self.index_last_injection = 0  # The fault list index of the last fault injected

    def generate_layer_probability(self, fault_list_length):
        """
        Helper function that generate a list of fault_list_length layers to be injected with a fault with a probability
        directly proportional to the number of parameters in that layer. The number of occurrences of a layer in the
        list is equal to the fault to be injected in that layer.
        :param fault_list_length: The number of layers to select
        :return: A list of layers sampled with repetition from all the network layers.
        """

        # Compute the probability of each layer to be selected
        layer_parameters = [layer.count_params() for layer in self.network.layers]
        layer_probabilities = layer_parameters / np.sum(layer_parameters)

        # Sample with repetition from all the layers of the network
        target_layers = self.rng.choice(len(layer_probabilities),
                                        size=fault_list_length,
                                        replace=True,
                                        p=layer_probabilities)

        return target_layers

    def save_fault_list(self, folder_path):
        """
        Save the fault list as a pickle file
        :param folder_path: where to save the fault_list
        """

        if len(self.fault_list) == 0:
            raise EmptyFaultList('Impossible to save an empty fault list. Generate a fault list first.')

        with open(f'{folder_path}/fault_list_{self.seed:d}.pkl', 'wb') as file:
            pickle.dump(self.fault_list, file)

    def load_fault_list(self, folder_path):
        """
        Load a fault list saved from a pickle file in folder folder_path that uses the seed used when creating this
        class. The file_name must be formatted as fault_list_[{%09d}].pkl
        :param folder_path: path where the fault_list is saved
        :return:
        """
        with open(f'{folder_path}/fault_list_{self.seed:d}.pkl', 'rb') as file:
            self.fault_list = pickle.load(file)

    def inject_incremental_fault_with_function(self, increment_number, layer_fault_injection_function):
        """
        Inject new faults on top of those already present in the network. Fault injections are done layer by layer (i.e.
        we cycle trough all the layer and for each one of them we inject the corresponding fault from the incremental
        fault list). The update of the network weights is done once per layer using layer_fault_injection_function.
        :param increment_number: The number of faults to inject on top of those already present
        :param layer_fault_injection_function: The function called to perform the fault injection on the layer. Takes as
        parameters an array containing the weights of the layer, a list of fault to be injected in that layer and the
        number of faults to be injected
        """

        incremental_index = self.index_last_injection + increment_number
        if incremental_index > len(self.fault_list):
            raise OutOfFaultList('The index of the incremental is larger than the dimension of the fault list')

        print(f'Injecting {increment_number} faults')

        target_list = np.array(self.fault_list[self.index_last_injection: incremental_index])
        self.index_last_injection = incremental_index

        # Cycle trough all the selected layer
        for layer_info in zip(*np.unique(target_list[:, 0], return_counts=True)):
            layer_index = layer_info[0]  # Layer index
            layer_count = layer_info[1]  # How many fault to inject in this layer

            # Get the list of faults for the current layer
            fault_for_layer = target_list[np.where(target_list[:, 0] == layer_index)]

            weights = self.network.layers[layer_index].get_weights()[0]
            bias = self.network.layers[layer_index].get_weights()[1]

            # Call the function that inject layer_count faults in the current layer
            layer_fault_injection_function(weights, bias, fault_for_layer, layer_count)

            # Update the weight with the faulty value
            self.network.layers[layer_index].set_weights((weights, bias))

    def compute_increment(self, target_number):
        """
        Compute the number of fault needed to reach the target_number number of faults and throws an OutOfFaultList
        exception if it is not possible.
        :param target_number: The target number of faults required to be injected in the network.
        :return: The number of faults need to reach the target_number.
        """
        increment = target_number - self.index_last_injection

        if increment < 0:
            raise OutOfFaultList(
                'The number of fault desired is less than the number of faults already present in the network')
        return increment

    def generate_fault_list(self, fault_list_length):
        raise NotImplementedError()

    def inject_incremental_fault(self, increment_number):
        raise NotImplementedError()

    def inject_up_to(self, target_number):
        raise NotImplementedError()

    def fault_injection_campaign(self, number_of_faults, folder_path, fault_list_length=10000):
        """
        Perform a fault injection campaign for the current network, injecting up to number_of_faults faults. The fault
        list is generated if the corresponding pickle file it is not found in the folder_path.
        :param number_of_faults: how many fault to have in the network
        :param folder_path: path to the folder containing the pickle file, if it exists
        :param fault_list_length: the length of the fault list
        """

        # Check if the fault list is empty. If it is, load the fault list form a file if it exists, otherwise
        # generate it
        if len(self.fault_list) == 0:
            try:
                self.load_fault_list(folder_path)
            except FileNotFoundError:
                self.generate_fault_list(fault_list_length)
                self.save_fault_list(folder_path)

        # Inject the faults
        self.inject_up_to(number_of_faults)
