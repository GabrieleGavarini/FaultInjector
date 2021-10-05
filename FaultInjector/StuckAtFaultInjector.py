from FaultInjector.NetworkFaultInjector import NetworkFaultInjector
from FaultInjector.FaultInjectorEngine import FaultInjectorEngine

import numpy as np


class StuckAtFaultInjector(NetworkFaultInjector):

    def __init__(self, network, seed):
        super().__init__(network, seed)

    def generate_fault_list(self, fault_list_length=10000):
        """
        Generate and set the fault list. Each entry of the fault list is a list of three elements: the first element is
        the index of the layer, the second is a tuple containing the index of the weight, the last element is the bit to
        be flipped
        :param fault_list_length: Length of the fault list
        """

        target_layers = super().generate_layer_probability(fault_list_length)

        # For each layer selected, generate an injection index and a target bit
        for layer_index in target_layers:
            while True:
                # Get the probability of a fault affecting the bias versus the weights
                total_layer_params = self.network.layers[layer_index].get_weights()[0].size +\
                                     self.network.layers[layer_index].get_weights()[1].size
                weights_probability = self.network.layers[layer_index].get_weights()[0].size / total_layer_params
                # bias_or_weights = self.rng.choice([0, 1], p=[weights_probability, 1 - weights_probability])
                bias_or_weights = self.rng.choice([0, 1])
                # Where to inject the fault
                if bias_or_weights == 0:
                    injection_index = tuple([self.rng.integers(0, i) for i in self.layer_shape[layer_index]])
                else:
                    injection_index = tuple([self.rng.integers(self.network.layers[layer_index].get_weights()[1].size)])
                # Value to inject
                stuck_at_value = self.rng.integers(0, 1, endpoint=True)
                # Compose the fault details in a list
                fault_list_element = [layer_index, bias_or_weights, injection_index, self.rng.integers(0, 32), stuck_at_value]
                if fault_list_element not in self.fault_list:
                    break
            self.fault_list.append(fault_list_element)

    def inject_incremental_fault(self, increment_number):
        """
        Inject new faults on top of those already present in the network. Fault injections are done layer by layer (i.e.
        we cycle trough all the layer and for each one of them we inject the corresponding fault from the incremental
        fault list). The update of the network weights is done once per layer.
        :param increment_number: The number of faults to inject on top of those already present
        """
        def fault_injection(weights, bias, fault_list, layer_count):
            for i in np.arange(0, layer_count):
                # Get whether to inject a bias or a weight
                bias_or_weights = fault_list[i][1]
                # Get the index of the weight to inject
                injection_index = fault_list[i][2]
                # Get which bit to change
                injection_position = fault_list[i][3]
                # Get target value
                stuck_at_value = fault_list[i][4]
                # Perform the fault injection
                if bias_or_weights == 0:    # Inject into the weights
                    weights[injection_index] = FaultInjectorEngine.float32_stuck_at(float_number=weights[injection_index],
                                                                                    position=injection_position,
                                                                                    stuck_at_value=stuck_at_value)
                else:   # Inject into Biases
                    bias[injection_index] = FaultInjectorEngine.float32_stuck_at(float_number=bias[injection_index],
                                                                                 position=injection_position,
                                                                                 stuck_at_value=stuck_at_value)

        super().inject_incremental_fault_with_function(increment_number, fault_injection)

    def inject_up_to(self, target_number):
        """
        Inject as many fault as need in order to reach the target number of faults in the network
        :param target_number: Target number of fault to have in the network
        """

        increment = super().compute_increment(target_number)

        self.inject_incremental_fault(increment)

    # temp - Used for debugging
    def TEMP_load_fault_list(self, fault_list_location):

        with open(fault_list_location) as input_file:
            fault_list = input_file.read().splitlines()

            for fault in fault_list:
                fault_details = [int(x) for x in fault.split(' ')]
                self.fault_list.append([
                    fault_details[0],
                    [fault_details[6], fault_details[5], fault_details[3], fault_details[4]],
                    fault_details[2],
                    fault_details[1]])
                pass
