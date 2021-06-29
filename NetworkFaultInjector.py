import numpy as np
from FaultInjector import FaultInjector


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

        self.rng = np.random.default_rng(seed=self.seed)    # The dedicated random number generator

        self.layer_shape = [(layer.get_weights()[0].shape if (len(layer.get_weights()) > 0) else ())
                            for layer in self.network.layers]   # List of the shape of all layers

        self.fault_list = []           # The list containing all the fault generated for this run
        self.index_last_injection = 0  # The fault list index of the last fault injected

    def generate_bit_flip_fault_list(self, fault_list_length=10000):
        """
        Generate and set the fault list. Each entry of the fault list is a list of three elements: the first element is
        the index of the layer, the second is a tuple containing the index of the weight, the last element is the bit to
        be flipped
        :param fault_list_length: Length of the fault list
        """

        layer_parameters = [layer.count_params() for layer in self.network.layers]
        layer_probabilities = layer_parameters / np.sum(layer_parameters)

        # Generate the probability for the layer based on the number of parameters
        target_layers = self.rng.choice(len(layer_probabilities),
                                        size=fault_list_length,
                                        replace=True,
                                        p=layer_probabilities)

        # For each layer selected, generate an injection index and a target bit
        for layer_index in target_layers:
            injection_index = tuple([self.rng.integers(0, i) for i in self.layer_shape[layer_index]])
            self.fault_list.append([layer_index, injection_index, self.rng.integers(0, 32)])

    def bit_flip_increment(self, increment_number):
        """
        Inject new faults on top of those already present in the network. Fault injections are done layer by layer (i.e.
        we cycle trough all the layer and for each one of them we inject the corresponding fault from the incremental
        fault list). The update of the network weights is done once per layer.
        :param increment_number: The number of faults to inject on top of those already present
        :return:
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

            for i in np.arange(0, layer_count):
                # Get the index of the weight to inject
                injection_index = fault_for_layer[i][1]
                # Get which bit to flip
                injection_position = fault_for_layer[i][2]
                # Perform the fault injection
                weights[injection_index] = FaultInjector.float32_bit_flip(float_number=weights[injection_index],
                                                                          position=injection_position)

            # Update the weight with the faulty value
            self.network.layers[layer_index].set_weights((weights, bias))

    def bit_flip_up_to(self, target_number):
        """
        Inject as many fault as need in order to reach the target number of faults in the network
        :param target_number: Target number of fault to have in the network
        """

        increment = target_number - self.index_last_injection

        if increment < 0:
            raise OutOfFaultList('The number of fault desired is less than the number of faults already present in the network')

        self.bit_flip_increment(increment)
