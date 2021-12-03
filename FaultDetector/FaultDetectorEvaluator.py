import pandas as pd
from tqdm import tqdm

from Util.NoValidFormatException import NoValidFormatException


class FaultDetectorEvaluator:

    @staticmethod
    def confusion_matrix(actual_value, predicted_value):
        if actual_value:
            if predicted_value:
                return 'TP'
            else:
                return 'FN'
        else:
            if predicted_value:
                return 'FP'
            else:
                return 'TN'

    @staticmethod
    def evaluate_fault_detector(fault_detector_dict, run_sdc):
        """
        Evaluate the performance of the fault detector
        :param fault_detector_dict: A dict {image: fault_detected}
        :param run_sdc: A pandas dataframe containing the sdc-metric for a given run
        :return: a dataframe containing the confusion matrix for each image of the dataset
        """

        evaluation_df = pd.DataFrame(columns=['sdc-1', 'sdc-5', 'sdc-10%', 'sdc-20%'])
        results = {}

        for index, row in tqdm(run_sdc.iterrows(), total=len(run_sdc)):
            sdc_1 = FaultDetectorEvaluator.confusion_matrix(row['sdc-1'], fault_detector_dict[index])
            sdc_5 = FaultDetectorEvaluator.confusion_matrix(row['sdc-5'], fault_detector_dict[index])
            sdc_10_percent = FaultDetectorEvaluator.confusion_matrix(row['sdc-10%'], fault_detector_dict[index])
            sdc_20_percent = FaultDetectorEvaluator.confusion_matrix(row['sdc-20%'], fault_detector_dict[index])
            results[index] = {'sdc-1': sdc_1,
                              'sdc-5': sdc_5,
                              'sdc-10%': sdc_10_percent,
                              'sdc-20%': sdc_20_percent}

        evaluation_df['sdc-1'] = pd.Series({key: info['sdc-1'] for key, info in results.items()})
        evaluation_df['sdc-5'] = pd.Series({key: info['sdc-5'] for key, info in results.items()})
        evaluation_df['sdc-10%'] = pd.Series({key: info['sdc-10%'] for key, info in results.items()})
        evaluation_df['sdc-20%'] = pd.Series({key: info['sdc-20%'] for key, info in results.items()})

        return evaluation_df

    @staticmethod
    def evaluate_and_export_fault_detector(fault_detector_dict, run_sdc, file_name, output_dir, output_format='pickle'):
        """
        Evaluate the performance of the fault detector and saves it to the specified file.
        :param fault_detector_dict: A dict {image: fault_detected}
        :param run_sdc: A pandas dataframe containing the sdc-metric for a given run
        :param file_name: the name of the file to save
        :param output_dir: the directory where to save the file containing the dataframe
        :param output_format: either 'csv' or 'pickle'. The format in which to save the output file
        :return: a dataframe containing the confusion matrix for each image of the dataset
        """
        evaluation_df = FaultDetectorEvaluator.evaluate_fault_detector(fault_detector_dict, run_sdc)

        if output_format == 'pickle':
            evaluation_df.to_pickle(f'{output_dir}/{file_name}_performance.pkl')
        elif output_format == 'csv':
            evaluation_df.to_csv(f'{output_dir}/{file_name}_performance.csv')
        else:
            print('No valid format has been specified for saving the inference output')
            raise NoValidFormatException

        return evaluation_df

    @staticmethod
    def get_metrics(evaluation_df, value_if_zero=1, metric='sdc-1'):
        tp = len(evaluation_df[evaluation_df[metric] == 'TP'])
        fp = len(evaluation_df[evaluation_df[metric] == 'FP'])
        tn = len(evaluation_df[evaluation_df[metric] == 'TN'])
        fn = len(evaluation_df[evaluation_df[metric] == 'FN'])

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if not (tp + fp) == 0 else value_if_zero
        recall = tp / (tp + fn) if not (tp + fn) == 0 else value_if_zero

        fpr = fp / (fp + tn) if not (fp + tn) == 0 else value_if_zero
        tpr = 1 - fpr
        fnr = fn / (fn + tp) if not (fn + tp) == 0 else value_if_zero
        tnr = 1 - fnr

        return [accuracy,
                precision,
                recall,
                fpr,
                tpr,
                fnr,
                tnr]