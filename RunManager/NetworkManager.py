import os

from tqdm import tqdm

import numpy as np
import pandas as pd

from RunManager.ImageLoader import ImageLoader


class NetworkManager:

    def __init__(self, network, dataset_dir):
        self.network = network
        self.dataset_dir = dataset_dir

    def run_and_export_cvs(self, run_name, output_dir, pre_processing_function=None):

        vector_score_list = {}

        n = 0
        t = 0

        for label_index, dir_name in enumerate(tqdm(os.listdir(self.dataset_dir))):
            for image_index, file_name in enumerate(os.listdir(f'{self.dataset_dir}/{dir_name}')):

                loaded_image = ImageLoader.load_resize_center(f'{self.dataset_dir}/{dir_name}/{file_name}')

                if pre_processing_function is not None:
                    loaded_image = pre_processing_function(loaded_image)

                vector_score_list[file_name] = list(self.network.predict(np.expand_dims(loaded_image, axis=0))[0])

                prediction = np.argmax(vector_score_list[file_name])
                if prediction == label_index:
                    n += 1
                t += 1

        accuracy = n/t
        print(accuracy)

        df = pd.DataFrame.from_dict(vector_score_list, orient='index')
        df.to_csv(f'{output_dir}/{run_name}.csv')
