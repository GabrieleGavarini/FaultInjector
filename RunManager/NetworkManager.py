import os

from tqdm import tqdm

from scipy import special
import numpy as np
import pandas as pd

from RunManager.ImageLoader import ImageLoader
from Util.NoValidFormatException import NoValidFormatException


class NetworkManager:

    def __init__(self, network, dataset_dir, golden_results=None):
        self.network = network                              # The network used for inference
        self.dataset_dir = dataset_dir                      # The dataset used for inference

        self.clean_weights = self.network.get_weights()     # The weights of the network unaffected by faults
        
        self.current_results = None                         # A panda df containing the results of the current run
        self.golden_results = golden_results                # A panda df containing the results of a golden run. None if unavailable

    def reset_network(self):
        """
        Reset the network weights to their original values
        """
        self.network.set_weights(self.clean_weights)

    def compute_sdc_n(self, n, entry_name, top_1_index):
        """
        Compute the sdc-n
        :param n: How many value of the golden results to consider
        :param entry_name: the name of the entry to consider for the computation of the interval
        :param top_1_index: The index of the highest values element of the vector score of the current prediction
        (pre-softmax)
        :return: sdc-n
        """
        for n in range(1, n + 1):
            if self.golden_results[f'top_{n}_index'].loc[entry_name] == top_1_index:
                return False

        return True

    def compute_sdc_n_percent(self, n, entry_name, top_1, den):
        """
        Compute the sdc-n% metrics.
        :param n: Float between 0 and 1. How much the faulty output is from the golden output.
        :param entry_name: the name of the entry to consider for the computation of the interval
        :param top_1: The highest values element of the vector score of the current prediction (pre-softmax)
        :param den: The value of the denominator of the current prediction, used to compute the softmax
        :return: True if the top-1_f prediction of the faulty run belongs to the interval [top-1_g-n%: top-1_g+n%]
        """
        golden_top_1_after_sm = np.exp(self.golden_results.top_1.loc[entry_name]) / self.golden_results.den.loc[entry_name]
        try:
            current_top_1_after_sm = np.exp(top_1) / den
        except RuntimeWarning:
            current_top_1_after_sm = 0

        return ~((current_top_1_after_sm >= golden_top_1_after_sm - n) and (current_top_1_after_sm <= golden_top_1_after_sm + n))

    def compute_sdc_metrics(self, vector_score, entry_name):
        """
        Compute the sdc metrics, given a vector score
        :param vector_score: The vector score of the current run used to compute the metrics
        :param entry_name: The name of the entry to consider for the computation of the metrics
        :return:
        """
        # Compute sdc-1
        top_1_index = np.argsort(vector_score)[::-1][0]
        top_1 = vector_score[top_1_index]
        try:
            den = sum(np.exp(vector_score))
        except RuntimeWarning:
            den = np.inf

        sdc_metrics_entry = {'sdc-1': self.compute_sdc_n(1, entry_name, top_1_index),
                             'sdc-5': True,
                             'sdc-10%': True,
                             'sdc-20%': True}
        if not sdc_metrics_entry['sdc-1']:
            # If sdc-1 is false, also sdc-5 is false
            sdc_metrics_entry['sdc-5'] = False
            # Compute sdc-10%
            sdc_metrics_entry['sdc-10%'] = self.compute_sdc_n_percent(0.1,
                                                                      entry_name,
                                                                      top_1=top_1,
                                                                      den=den
                                                                      )
            if not sdc_metrics_entry['sdc-10%']:
                # If sdc-10% is ture, then also sdc-20% is true
                sdc_metrics_entry['sdc-20%'] = False
            else:
                # Compute sdc-20%
                sdc_metrics_entry['sdc-20%'] = self.compute_sdc_n_percent(0.2,
                                                                          entry_name,
                                                                          top_1=top_1,
                                                                          den=den
                                                                          )
        # If sdc-1 is true, sdc-n% are meaningless
        else:
            # Compute sdc-5
            sdc_metrics_entry['sdc-5'] = self.compute_sdc_n(5, entry_name, top_1_index)
        return sdc_metrics_entry

    def run_and_export(self, run_name, output_dir, output_format='pickle', top_n=None, open_max_activation_vectors=None,
                       compute_sdc_metrics=False, pre_processing_function=None):
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
        :param compute_sdc_metrics: Whether or not to include the sdc metrics into the dataframe. Throws an exception if
        the golden run results are not defined.
        :param pre_processing_function: The pre-processing function to apply to the input dataset.
        :return returns the dataframe saved in the specified file
        """

        # Remove the previous results, if present
        self.current_results = None

        vector_score_list = {}
        open_max_distance = {}
        sdc_metrics = {}

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
                    den = sum(np.exp(vector_score))
                    top_n_indexes = np.argsort(vector_score)[::-1][:top_n]
                    top_n_values = [vector_score[i] for i in top_n_indexes]
                    vector_score_list[file_name] = list(top_n_values) + list(top_n_indexes) + list([den])

                if open_max_activation_vectors is not None:
                    # Compute the distance between the mav and the vector score (computed after the softmax)
                    distance = open_max_activation_vectors[dir_name] - special.softmax(vector_score)
                    open_max_distance[file_name] = np.linalg.norm(distance)

                if compute_sdc_metrics:
                    sdc_metrics[file_name] = self.compute_sdc_metrics(vector_score, file_name)

        # Save the results as a whole or only the top-n
        if top_n is None:
            self.current_results = pd.DataFrame.from_dict(vector_score_list, orient='index')
        else:
            columns = [f'top_{i}' for i in range(1, top_n + 1)] +\
                      [f'top_{i}_index' for i in range(1, top_n + 1)] +\
                      ['den']
            self.current_results = pd.DataFrame.from_dict(vector_score_list, orient='index', columns=columns)

        # Compute the distance from the mav
        if open_max_activation_vectors is not None:
            self.current_results['OpenMaxDistance'] = pd.Series(open_max_distance)

        # Compute the sdc metrics
        if compute_sdc_metrics:
            self.current_results['sdc-1'] = pd.Series({key: info['sdc-1'] for key, info in sdc_metrics.items()})
            self.current_results['sdc-5'] = pd.Series({key: info['sdc-5'] for key, info in sdc_metrics.items()})
            self.current_results['sdc-10%'] = pd.Series({key: info['sdc-10%'] for key, info in sdc_metrics.items()})
            self.current_results['sdc-20%'] = pd.Series({key: info['sdc-20%'] for key, info in sdc_metrics.items()})

        if output_format == 'pickle':
            self.current_results.to_pickle(f'{output_dir}/{run_name}_inference_result.pkl')
        elif output_format == 'csv':
            self.current_results.to_csv(f'{output_dir}/{run_name}_inference_result.csv')
        else:
            print('No valid format has been specified for saving the inference output')
            raise NoValidFormatException

        return self.current_results

    def save_golden_results(self):
        """
        Save the current run results as the golden run results
        """
        self.golden_results = self.current_results
