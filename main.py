import sys

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from FaultInjector.StuckAtFaultInjector import StuckAtFaultInjector
from RunManager.NetworkManager import NetworkManager
from RunManager.Baseline import Baseline

if __name__ == "__main__":

    # Testing_dir: The folder where the testing dataset using to perform the inference. This is used to compute how much
    #              a fault impacts a run
    # Detection_dir: The folder containing the detection dataset used to compute the metrics useful for the fault
    #                detectors.
    testing_dir = sys.argv[1]
    detection_dir = sys.argv[2]

    # mav_dir: where to load/save the file containing the mean activation vector for the detection dataset.
    mav_file_location = sys.argv[3]

    batch_size = 128

    top_n = 5
    output_format = 'pickle'
    # number_of_faults_list = [100, 200, 500]
    number_of_faults_list = [100]

    seed_list = [113, 127]

    # STEP 1 - Golden Run
    # 1.1 - Create the network
    vgg = VGG16(classifier_activation=None)
    vgg.compile(metrics=['accuracy'])
    # 1.2 - Initialize the network manager
    network_manager = NetworkManager(network=vgg, dataset_dir=testing_dir)

    baseline = Baseline(network=vgg, dataset_dir=detection_dir)
    open_max_activation_vectors = baseline.compute_mean_activation_vectors(file_location=mav_file_location,
                                                                           pre_processing_function=preprocess_input)

    # 1.3 - Execute the golden run
    network_manager.run_and_export(run_name=f'vgg_imagenet',
                                   output_dir='GoldenRunResults',
                                   top_n=top_n,
                                   open_max_activation_vectors=open_max_activation_vectors,
                                   output_format=output_format,
                                   pre_processing_function=preprocess_input)
    network_manager.save_golden_results()

    # STEP 2 - Faulty Run
    for seed in seed_list:
        # 2.1 - reset the network to its original state (i.e. load the original weights)
        network_manager.reset_network()
        # 2.2 - Initialize the fault injector
        fault_injector = StuckAtFaultInjector(vgg, seed)
        # 2.3 - Generate a fault list and perform a fault injection campaign for the number of faults in the list
        for number_of_faults in number_of_faults_list:
            fault_injector.fault_injection_campaign(number_of_faults=number_of_faults,
                                                    folder_path='FaultList',
                                                    fault_list_length=100)
            # 2.4 - Execute a faulty run
            network_manager.run_and_export(run_name=f'vgg_imagenet_{seed}_{number_of_faults}',
                                           output_dir='FaultyRunResults',
                                           top_n=top_n,
                                           open_max_activation_vectors=open_max_activation_vectors,
                                           compute_sdc_metrics=True,
                                           output_format=output_format,
                                           pre_processing_function=preprocess_input)
