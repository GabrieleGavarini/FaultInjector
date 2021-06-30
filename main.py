import sys
import os

from tqdm import tqdm

import numpy as np
import pandas as pd

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from ImageLoader import ImageLoader
from NetworkFaultInjector import NetworkFaultInjector


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

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    seed = 11234
    batch_size = 128

    vgg = VGG16()
    vgg.compile(metrics=['accuracy'])

    fault_injector = NetworkFaultInjector(vgg, seed)
    fault_injector.generate_bit_flip_fault_list()
    fault_injector.bit_flip_up_to(100)

    vector_score_list = []

    accuracy = 0

    for label_index, dir_name in enumerate(tqdm(os.listdir(input_dir))):
        for image_index, file_name in enumerate(os.listdir(f'{input_dir}/{dir_name}')):

            loaded_image = ImageLoader.load_resize_center(f'{input_dir}/{dir_name}/{file_name}')
            loaded_image = preprocess_input(loaded_image)

            vector_score = vgg.predict(np.expand_dims(loaded_image, axis=0))
            prediction = np.argmax(vector_score)

            if prediction == label_index:
                accuracy = accuracy + 1

            vector_score_list.append(vector_score[0])

            # print(f'Target: {label_index}, Predicted: {prediction}')

    print(f'Accuracy of {accuracy/10000}%')

    df = pd.DataFrame(np.array(vector_score_list))
    df.to_csv(f'{output_dir}/run_{seed}.csv')
