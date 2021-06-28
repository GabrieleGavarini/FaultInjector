import numpy as np
from FaultInjector import FaultInjector


class NetworkFaultInjector:

    def __init__(self, network, seed):
        self.network = network
        self.seed = seed

        self.rng = np.random.default_rng(seed=self.seed)

    def bit_flip_injection_campaign(self, number_faults):
        """
        Run a fault injection campaign
        :param number_faults: The number of faults to inject
        :return:
        """
        layer_parameters = [layer.count_params() for layer in self.network.layers]
        layer_probabilities = layer_parameters / np.sum(layer_parameters)

        # Generate the probability for the layer based on the number of parameters
        target_layers = self.rng.choice(len(layer_probabilities),
                                        size=number_faults,
                                        replace=True,
                                        p=layer_probabilities)

        # Cycle trough all the selected layer
        for layer_info in zip(*np.unique(target_layers, return_counts=True)):
            layer_index = layer_info[0]     # Layer index
            layer_count = layer_info[1]     # How many fault to inject in this layer

            layer = self.network.layers[layer_index]

            weights = layer.get_weights()[0]
            # TODO: inject also on bias
            bias = layer.get_weights()[1]

            for _ in np.arange(0, layer_count):
                # Get the index of the weight to inject
                injection_index = tuple([self.rng.integers(0, i) for i in weights.shape])
                # Perform the fault injection
                weights[injection_index] = FaultInjector.float32_bit_flip(float_number=weights[injection_index],
                                                                          position=self.rng.integers(0, 32))

            # Update the weight with the faulty value
            self.network.layers[layer_index].set_weights((weights, bias))
