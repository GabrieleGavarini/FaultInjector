import sys

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from FaultInjector.StuckAtFaultInjector import StuckAtFaultInjector
from RunManager.NetworkManager import NetworkManager

if __name__ == "__main__":

    input_dir = sys.argv[1]     # Location of the dataset
    output_dir = sys.argv[2]    # Where to store the dataframe
    batch_size = 128

    seed_1 = 113
    seed_2 = 127

    vgg_1 = VGG16()
    vgg_1.compile(metrics=['accuracy'])
    network_manager_1 = NetworkManager(network=vgg_1, dataset_dir=input_dir)
    fault_injector_1 = StuckAtFaultInjector(vgg_1, seed_1)
    fault_injector_1.fault_injection_campaign(1000, 'FaultList')
    network_manager_1.run_and_export_cvs(run_name=f'vgg_imagenet_{seed_1}_inference_result.csv',
                                         output_dir='FaultyRunResults',
                                         pre_processing_function=preprocess_input)

    vgg_2 = VGG16()
    vgg_2.compile(metrics=['accuracy'])
    network_manager_2 = NetworkManager(network=vgg_2, dataset_dir=input_dir)
    fault_injector_2 = StuckAtFaultInjector(vgg_2, seed_2)
    fault_injector_2.fault_injection_campaign(1000, 'FaultList')
    network_manager_2.run_and_export_cvs(run_name=f'vgg_imagenet_{seed_2}_inference_result.csv',
                                         output_dir='FaultyRunResults',
                                         pre_processing_function=preprocess_input)
