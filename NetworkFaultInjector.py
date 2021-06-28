import tensorflow as tf
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

        for layer in [self.network.layers[i] for i in target_layers]:
            weights = layer.get_weights()[0]
            # TODO: inject also on bias
            bias = layer.get_weights()[1]

            # Get the index of the weight to inject
            index = tuple([self.rng.integers(0, i) for i in weights.shape])
            # Perform the fault injection
            weights[index] = FaultInjector.float32_bit_flip(weights[index], self.rng.integers(0, 32))

            # Update the weight with the faulty value
            layer.set_weights((weights, bias))
