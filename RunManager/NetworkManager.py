import os

from tqdm import tqdm

import numpy as np
import pandas as pd

from RunManager.ImageLoader import ImageLoader


class NetworkManager:

    def __init__(self, network, dataset_dir):
        self.network = network
        self.dataset_dir = dataset_dir

    def run_and_export_cvs(self, run_name, output_dir='../FaultyRunResults', pre_processing_function=None):

        vector_score_list = []

        for label_index, dir_name in enumerate(tqdm(os.listdir(self.dataset_dir))):
            for image_index, file_name in enumerate(os.listdir(f'{self.dataset_dir}/{dir_name}')):

                loaded_image = ImageLoader.load_resize_center(f'{self.dataset_dir}/{dir_name}/{file_name}')

                if pre_processing_function is not None:
                    loaded_image = pre_processing_function(loaded_image)

                vector_score = self.network.predict(np.expand_dims(loaded_image, axis=0))
                prediction = np.argmax(vector_score)

                vector_score_list.append(vector_score[0])

                # print(f'Target: {label_index}, Predicted: {prediction}')

        df = pd.DataFrame(np.array(vector_score_list))
        df.to_csv(f'{output_dir}/{run_name}.csv')
