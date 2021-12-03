import os

from tqdm import tqdm

import numpy as np
import pickle

from RunManager.ImageLoader import ImageLoader


class FaultDetectorMetrics:

    def __init__(self, network, dataset_dir):
        self.network = network
        self.dataset_dir = dataset_dir

    def compute_score_based_threshold(self, pre_processing_function=None, file_location=None):
        """
        Compute the score based threshold on the specified dataset given the specified pre-processing function. If file
        location is specified it tries to load the mean activation vector from the file. If the file does not exists it
        first computes the threshold and then it saves to the specified location.
        :param pre_processing_function: Pre-processing function to apply to the dataset.
        :param file_location: Either None or a string. If not None, it is the location where to save/load the file
        containing the threshold.
        :return: the threshold
        """

        try:
            threshold = pickle.load(open(file_location, 'rb'))
            return threshold
        except (OSError, IOError):
            threshold = [0, 10e6]

            for label_index, dir_name in enumerate(tqdm(os.listdir(self.dataset_dir))):
                for image_index, file_name in enumerate(os.listdir(f'{self.dataset_dir}/{dir_name}')):

                    # Load the image, resize it and perform a center crop
                    loaded_image = ImageLoader.load_resize_center(f'{self.dataset_dir}/{dir_name}/{file_name}')

                    # Pre-process the image if a function has been specified
                    if pre_processing_function is not None:
                        loaded_image = pre_processing_function(loaded_image)

                    # Compute the vector score
                    vector_score = list(self.network.predict(np.expand_dims(loaded_image, axis=0))[0])

                    # If the maximum of the vector score is bigger than the previous threshold update the threshold
                    threshold = [max(threshold[0], max(vector_score)), min(threshold[1], max(vector_score))]

            pickle.dump(threshold, open(file_location, 'wb'))
            return threshold

    def compute_mean_activation_vectors(self, pre_processing_function=None, file_location=None):
        """
        Compute the open-max mean activation vector (MAV) given the specified pre-processing function. If file location
        is specified it tries to load the mean activation vector from the file. If the file does not exists it first
        computes the MAV and then it saves to the specified location.
        :param pre_processing_function: Pre-processing function to apply to the dataset.
        :param file_location: Either None or a string. If not None, it is the location where to save/load the file
        containing the MAV.
        :return: the MAV for all the classes.
        """

        # Initialize the dictionary containing the mean activation vectors
        try:
            mean_activation_vectors = pickle.load(open(file_location, 'rb'))
            return mean_activation_vectors
        except (OSError, IOError):
            mean_activation_vectors = {}

            for label_index, dir_name in enumerate(tqdm(os.listdir(self.dataset_dir))):

                mean_activation_vector_sum = np.zeros(len(os.listdir(self.dataset_dir)))

                for image_index, file_name in enumerate(os.listdir(f'{self.dataset_dir}/{dir_name}')):

                    # Load the image, resize it and perform a center crop
                    loaded_image = ImageLoader.load_resize_center(f'{self.dataset_dir}/{dir_name}/{file_name}')

                    # Pre-process the image if a function has been specified
                    if pre_processing_function is not None:
                        loaded_image = pre_processing_function(loaded_image)

                    # Compute the vector score
                    vector_score = list(self.network.predict(np.expand_dims(loaded_image, axis=0))[0])

                    # Add the result to the cumulative sum
                    mean_activation_vector_sum = mean_activation_vector_sum + vector_score

                # Compute the average
                vector_count = len(os.listdir(f'{self.dataset_dir}/{dir_name}'))
                mean_activation_vectors[f'{dir_name}'] = mean_activation_vector_sum / vector_count

            pickle.dump(mean_activation_vectors, open(file_location, 'wb'))
            return mean_activation_vectors

    def compute_mav_distance_threshold(self, mav, pre_processing_function=None, file_location=None):
        """
        Compute the distance from the mav for the given dataset.
        :param mav: the mean activation vectors
        :param pre_processing_function: Pre-processing function to apply to the dataset.
        :param file_location: Either None or a string. If not None, it is the location where to save/load the file
        containing the MAV threshold.
        :return:
        """

        try:
            threshold = pickle.load(open(file_location, 'rb'))
            return threshold
        except (OSError, IOError):
            threshold = 0

            for label_index, dir_name in enumerate(tqdm(os.listdir(self.dataset_dir))):
                for image_index, file_name in enumerate(os.listdir(f'{self.dataset_dir}/{dir_name}')):

                    # Load the image, resize it and perform a center crop
                    loaded_image = ImageLoader.load_resize_center(f'{self.dataset_dir}/{dir_name}/{file_name}')

                    # Pre-process the image if a function has been specified
                    if pre_processing_function is not None:
                        loaded_image = pre_processing_function(loaded_image)

                    # Compute the vector score
                    vector_score = list(self.network.predict(np.expand_dims(loaded_image, axis=0))[0])

                    # If the maximum of the vector score is bigger than the previous threshold update the threshold
                    distance = np.linalg.norm(mav[f'{dir_name}'] - vector_score)
                    threshold = max(threshold, distance)

            pickle.dump(threshold, open(file_location, 'wb'))
            return threshold
