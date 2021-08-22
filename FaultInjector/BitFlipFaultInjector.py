from FaultInjector.NetworkFaultInjector import NetworkFaultInjector
from FaultInjector.FaultInjectorEngine import FaultInjectorEngine

import numpy as np


class BitFlipFaultInjector(NetworkFaultInjector):

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
                injection_index = tuple([self.rng.integers(0, i) for i in self.layer_shape[layer_index]])
                fault_list_element = [layer_index, injection_index, self.rng.integers(0, 32)]
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
        def fault_injection(weights, fault_list, layer_count):
            for i in np.arange(0, layer_count):
                # Get the index of the weight to inject
                injection_index = fault_list[i][1]
                # Get which bit to flip
                injection_position = fault_list[i][2]
                # Perform the fault injection
                weights[injection_index] = FaultInjectorEngine.float32_bit_flip(float_number=weights[injection_index],
                                                                                position=injection_position)

        super().inject_incremental_fault_with_function(increment_number, fault_injection)

    def inject_up_to(self, target_number):
        """
        Inject as many fault as need in order to reach the target number of faults in the network
        :param target_number: Target number of fault to have in the network
        """

        increment = super().compute_increment(target_number)

        self.inject_incremental_fault(increment)
