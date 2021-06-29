from tensorflow.keras.applications import VGG16
from NetworkFaultInjector import NetworkFaultInjector

if __name__ == "__main__":
    vgg = VGG16()

    fault_injector = NetworkFaultInjector(vgg, 11234)
    fault_injector.generate_bit_flip_fault_list()

    first_fault = fault_injector.fault_list[0]
    print(f'Value before the injection: {vgg.layers[first_fault[0]].get_weights()[0][first_fault[1]]}')
    fault_injector.bit_flip_increment(1000)
    print(f'Value after the injection: {vgg.layers[first_fault[0]].get_weights()[0][first_fault[1]]}')
