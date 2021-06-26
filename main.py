import tensorflow as tf
from tensorflow.keras.applications import VGG16

from faultInjector import float32_bit_flip, float32_stuck_at

vgg = VGG16()

weights = vgg.layers[1].get_weights()
# weights[0] -> weights
#   HxWxIxC
# weights[1] -> bias

# [W\B][H][W][I][C]
bit_flip_target = weights[0][1][1][1][1]
print(bit_flip_target)
print(float32_bit_flip(bit_flip_target, 30))
print(float32_stuck_at(bit_flip_target, 29, 0))
print(float32_stuck_at(bit_flip_target, 29, 1))
print(float32_stuck_at(bit_flip_target, 31, 1))
print(float32_stuck_at(bit_flip_target, 31, 0))

weights[0][1][1][1][1] = float32_bit_flip(weights[0][1][1][1][1], 31)
vgg.layers[1].set_weights(weights)

new_weights = vgg.layers[1].get_weights()

print(new_weights[0][1][1][1][1])
