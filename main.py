import sys

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from FaultInjector.NetworkFaultInjector import NetworkFaultInjector
from RunManager.NetworkManager import NetworkManager


def fault_injection_test_case():
    fault_0 = fault_injector.fault_list[0]  # Should change after 1st injection
    fault_1000 = fault_injector.fault_list[1000]  # Should change after 2nd injection
    fault_1999 = fault_injector.fault_list[1999]  # Should change after 2nd injection
    fault_2000 = fault_injector.fault_list[2000]  # Should not change

    print(f'Value of 0 before the injection: {vgg.layers[fault_0[0]].get_weights()[0][fault_0[1]]}')
    print(f'Value of 1000 before the injection: {vgg.layers[fault_1000[0]].get_weights()[0][fault_1000[1]]}')
    print(f'Value of 1999 before the injection: {vgg.layers[fault_1999[0]].get_weights()[0][fault_1999[1]]}')
    print(f'Value of 2000 before the injection: {vgg.layers[fault_2000[0]].get_weights()[0][fault_2000[1]]}')
    print('\n')

    fault_injector.bit_flip_increment(1000)
    print(f'Value of 0 after injection #{1}: {vgg.layers[fault_0[0]].get_weights()[0][fault_0[1]]}')
    print(f'Value of 1000 after injection #{1}: {vgg.layers[fault_1000[0]].get_weights()[0][fault_1000[1]]}')
    print(f'Value of 1999 after injection #{1}: {vgg.layers[fault_1999[0]].get_weights()[0][fault_1999[1]]}')
    print(f'Value of 2000 after injection #{1}: {vgg.layers[fault_2000[0]].get_weights()[0][fault_2000[1]]}')
    print('\n')

    fault_injector.bit_flip_up_to(2000)
    print(f'Value of 0 after injection #{2}: {vgg.layers[fault_0[0]].get_weights()[0][fault_0[1]]}')
    print(f'Value of 1000 after injection #{2}: {vgg.layers[fault_1000[0]].get_weights()[0][fault_1000[1]]}')
    print(f'Value of 1999 after injection #{2}: {vgg.layers[fault_1999[0]].get_weights()[0][fault_1999[1]]}')
    print(f'Value of 2000 after injection #{2}: {vgg.layers[fault_2000[0]].get_weights()[0][fault_2000[1]]}')
    print('\n')


if __name__ == "__main__":

    input_dir = sys.argv[1]     # Location of the dataset
    output_dir = sys.argv[2]    # Where to store the dataframe
    seed = 113
    batch_size = 128

    vgg = VGG16()
    vgg.compile(metrics=['accuracy'])

    network_manager = NetworkManager(network=vgg, dataset_dir=input_dir)

    fault_injector = NetworkFaultInjector(vgg, seed)
    fault_injector.generate_bit_flip_fault_list()
    # fault_injector.bit_flip_up_to(100)

    network_manager.run_and_export_cvs(run_name=f'run_{seed}',
                                       output_dir=output_dir,
                                       pre_processing_function=preprocess_input)

