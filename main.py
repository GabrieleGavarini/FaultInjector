import sys

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from FaultInjector.StuckAtFaultInjector import StuckAtFaultInjector
from RunManager.NetworkManager import NetworkManager

if __name__ == "__main__":

    input_dir = sys.argv[1]  # Location of the dataset
    batch_size = 128

    top_n = 5
    output_format = 'pickle'
    number_of_faults = 1000

    seed_list = [113, 127]

    # STEP 1 - Golden Run
    # 1.1 - Create the network
    vgg = VGG16(classifier_activation=None)
    vgg.compile(metrics=['accuracy'])
    # 1.2 - Initialize the network manager
    network_manager = NetworkManager(network=vgg, dataset_dir=input_dir)
    # 1.3 - Execute the golden run
    # network_manager.run_and_export(run_name=f'vgg_imagenet_inference_result',
    #                                output_dir='GoldenRunResults',
    #                                top_n=top_n,
    #                                output_format=output_format,
    #                                pre_processing_function=preprocess_input)

    # STEP 2 - Faulty Run
    for seed in seed_list:
        # 2.1 - reset the network to its original state (i.e. load the original weights)
        network_manager.reset_network()
        # 2.2 - Initialize the fault injector
        fault_injector = StuckAtFaultInjector(vgg, seed)
        # 2.3 - Generate a fault list and perform a fault injection campaign
        fault_injector.fault_injection_campaign(number_of_faults=number_of_faults, folder_path='FaultList', fault_list_length=100000)
        # 2.4 - Execute a faulty run
        network_manager.run_and_export(run_name=f'vgg_imagenet_{seed}_{number_of_faults}_inference_result',
                                       output_dir='FaultyRunResults',
                                       top_n=top_n,
                                       output_format=output_format,
                                       pre_processing_function=preprocess_input)
