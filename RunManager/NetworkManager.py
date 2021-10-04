import os

from tqdm import tqdm

import numpy as np
import pandas as pd

from RunManager.ImageLoader import ImageLoader


class NoValidFormatException(Exception):
    pass


class NetworkManager:

    def __init__(self, network, dataset_dir):
        self.network = network
        self.dataset_dir = dataset_dir

        self.clean_weights = self.network.get_weights()

    def reset_network(self):
        """
        Reset the network weights to their original values
        """
        self.network.set_weights(self.clean_weights)

    def run_and_export(self, run_name, output_dir, output_format='pickle', top_n=None, open_max_activation_vectors=None,
                       pre_processing_function=None):
        """
        Run an inference of the network for the stored dataset, applying the specified pre-processing function. The run
        is then saved in the specified format in output_dir. If top_n is specified only the top_n values and relative
        indexes are saved.
        :param run_name: The name of the run, used to name the file.
        :param output_dir: Where to save the run.
        :param output_format: One of ['pickle', 'csv']. The format used to save the results.
        :param top_n: Either None or an integer. If no value is specified, the saved dataframe contains a column for
        each element of the vector score. If an integer n is specified, the dataframe contains 2*n columns, where the
        first n columns contain the value of the top n values and the other contains the related n indexes.
        :param open_max_activation_vectors: Either None or a dictionary of vectors. The mean activation vectors of all
        the classes. The dictionary entry is a pair key data where the key is the class name and the data is the mean
        activation vector for that class.
        :param pre_processing_function: The pre-processing function to apply to the input dataset.
        """
        vector_score_list = {}
        open_max_distance = {}

        for label_index, dir_name in enumerate(tqdm(os.listdir(self.dataset_dir))):
            for image_index, file_name in enumerate(os.listdir(f'{self.dataset_dir}/{dir_name}')):

                # Load the image, resize it and perform a center crop
                loaded_image = ImageLoader.load_resize_center(f'{self.dataset_dir}/{dir_name}/{file_name}')

                # Pre-process the image if a function has been specified
                if pre_processing_function is not None:
                    loaded_image = pre_processing_function(loaded_image)

                # Compute the vector score
                vector_score = list(self.network.predict(np.expand_dims(loaded_image, axis=0))[0])
                # Create a new entry in the dictionary containing the prediction
                if top_n is None:
                    vector_score_list[file_name] = vector_score
                else:
                    top_n_indexes = np.argsort(vector_score)[::-1][:top_n]
                    top_n_values = [vector_score[i] for i in top_n_indexes]
                    vector_score_list[file_name] = list(top_n_values) + list(top_n_indexes)

                if open_max_activation_vectors is not None:
                    distance = open_max_activation_vectors[dir_name] - vector_score
                    open_max_distance[file_name] = np.linalg.norm(distance)

        if top_n is None:
            df = pd.DataFrame.from_dict(vector_score_list, orient='index')
        else:
            columns = [f'top_{i}' for i in range(1, top_n + 1)] + [f'top_{i}_index' for i in range(1, top_n + 1)]
            df = pd.DataFrame.from_dict(vector_score_list, orient='index', columns=columns)

        if open_max_activation_vectors is not None:
            df['OpenMaxDistance'] = open_max_distance.values()

        if output_format is 'pickle':
            df.to_pickle(f'{output_dir}/{run_name}.pkl')
        elif output_format is 'csv':
            df.to_csv(f'{output_dir}/{run_name}.csv')
        else:
            print('No valid format has been specified for saving the inference output')
            raise NoValidFormatException
