from tensorflow.keras.applications import VGG16
from NetworkFaultInjector import NetworkFaultInjector

if __name__ == "__main__":
    vgg = VGG16()

    fault_injector = NetworkFaultInjector(vgg, 11234)
    fault_injector.generate_bit_flip_fault_list()

    fault_0 = fault_injector.fault_list[0]          # Should change after 1st injection
    fault_1000 = fault_injector.fault_list[1000]    # Should change after 2nd injection
    fault_1999 = fault_injector.fault_list[1999]    # Should change after 2nd injection
    fault_2000 = fault_injector.fault_list[2000]    # Should not change

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
