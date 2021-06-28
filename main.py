import tensorflow as tf
from tensorflow.keras.applications import VGG16
from NetworkFaultInjector import NetworkFaultInjector


vgg = VGG16()

fault_injector = NetworkFaultInjector(vgg, 11234)
fault_injector.bit_flip_injection_campaign(10000)

