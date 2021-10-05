import pandas as pd

class FaultDetectorEvaluator:

    @staticmethod
    def confusion_matrix(self, actual_value, predicted_value):
        if actual_value == predicted_value == True:
            return 'TP'
        elif actual_value == predicted_value == False:
            return 'TN'
        elif actual_value is True:
            return 'FN'
        else:
            return 'TN'

    @staticmethod
    def evaluate_fault_detector(fault_detector_dict, run_sdc):
        """
        Evaluate the performance of the fault detector
        :param fault_detector_dict: A dict {image: fault_detected}
        :param run_sdc: A pandas dataframe containing the sdc-metric for a given run
        :return:
        """

        df = pd.DataFrame(columns=['sdc-1', 'sdc-5', 'sdc-10%', 'sdc-20%'])
        results = {}

        for index, row in run_sdc.iterrows():
            sdc_1 = FaultDetectorEvaluator.confusion_matrix(row['sdc-1'], fault_detector_dict[index])
            sdc_5 = FaultDetectorEvaluator.confusion_matrix(row['sdc-5'], fault_detector_dict[index])
            sdc_10_perc = FaultDetectorEvaluator.confusion_matrix(row['sdc-10%'], fault_detector_dict[index])
            sdc_20_perc = FaultDetectorEvaluator.confusion_matrix(row['sdc-20%'], fault_detector_dict[index])

        df['sdc-1'] = pd.Series({key: info['sdc-1'] for key, info in results.items()})
        df['sdc-5'] = pd.Series({key: info['sdc-5'] for key, info in results.items()})
        df['sdc-10%'] = pd.Series({key: info['sdc-10%'] for key, info in results.items()})
        df['sdc-20%'] = pd.Series({key: info['sdc-20%'] for key, info in results.items()})

        return df
