import sys
import numpy as np
import pickle
import os
import re
import csv
from tqdm import tqdm
import pandas as pd

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from RunManager.NetworkManager import NetworkManager

from FaultDetector.FaultDetectorMetrics import FaultDetectorMetrics
from FaultDetector.ScoreBasedFaultDetector import ScoreBasedFaultDetector
from FaultDetector.MavFaultDetector import MavFaultDetector
from FaultDetector.FaultDetectorEvaluator import FaultDetectorEvaluator

# Testing_dir: The folder where the testing dataset using to perform the inference. This is used to compute how much
#              a fault impacts a run
# Detection_dir: The folder containing the detection dataset used to compute the metrics useful for the fault
#                detectors.
testing_dir = None
detection_dir = sys.argv[1]

# mav_dir: where to load/save the file containing the mean activation vector for the detection dataset.
# threshold_dir: where to load/save the file containing the threshold for the detection dataset.
mav_file_location = sys.argv[2]
threshold_file_location = sys.argv[3]

vgg = VGG16(classifier_activation=None)
vgg.compile(metrics=['accuracy'])

# 1.3 - Initialize the fault detector metrics
metrics = FaultDetectorMetrics(network=vgg, dataset_dir=detection_dir)

mode = 'ScoreBased'
# mode = 'ScoreBased'
metric = 'sdc-5'
gamma_list = [0.5]

if mode == 'OpenMax':
    open_max_activation_vectors = metrics.compute_mean_activation_vectors(file_location=f'{mav_file_location}/mav.pkl',
                                                                          pre_processing_function=preprocess_input)
    initial_threshold = metrics.compute_mav_distance_threshold(mav=open_max_activation_vectors,
                                                               file_location=f'{mav_file_location}/distance.pkl',
                                                               pre_processing_function=preprocess_input)
    filename = f'FaultDetectorResults/OpenMax/{metric}.csv'
else:
    initial_threshold = metrics.compute_score_based_threshold(file_location=f'{threshold_file_location}/threshold.pkl',
                                                              pre_processing_function=preprocess_input)
    filename = f'FaultDetectorResults/ScoreBased/{metric}.csv'

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    header = ['Gamma', 'Accuracy', 'Precision', 'Recall', 'FPR', 'TPR', 'FNR', 'TNR']
    writer.writerow(header)
    f.flush()

df_list = []
for file_location in tqdm(os.listdir('FaultyRunResults')):
    seed = re.findall(r'[0-9]+', file_location)[0]
    number_of_faults = re.findall(r'[0-9]+', file_location)[1]
    inference_results = pickle.load(open(f'FaultyRunResults/{file_location}', 'rb'))

    inference_results['Seed'] = seed
    inference_results['FaultNumber'] = number_of_faults

    df_list.append(inference_results)

inference_results = pd.concat(df_list, ignore_index=True)

for gamma in gamma_list:

    print(f'Evaluating {mode} for gamma={gamma}', flush=True)
    threshold = gamma * initial_threshold

    if mode == 'OpenMax':
        mav_fault_detector = MavFaultDetector(inference_result=inference_results,
                                              threshold=threshold)
        mav_fault_detector_results = mav_fault_detector.detect_faults()
        mav_evaluation = FaultDetectorEvaluator.evaluate_and_export_fault_detector(
            fault_detector_dict=mav_fault_detector_results,
            run_sdc=inference_results,
            file_name=f'vgg_imagenet_mav',
            output_dir='FaultDetectorResults')
        results = FaultDetectorEvaluator.get_metrics(mav_evaluation, metric=metric)
    else:
        score_based_fault_detector = ScoreBasedFaultDetector(inference_result=inference_results,
                                                             threshold=threshold)
        score_based_fault_detector_results = score_based_fault_detector.detect_faults()
        score_based_evaluation = FaultDetectorEvaluator.evaluate_and_export_fault_detector(
            fault_detector_dict=score_based_fault_detector_results,
            run_sdc=inference_results,
            file_name=f'vgg_imagenet_score_based',
            output_dir='FaultDetectorResults')
        score_based_results = FaultDetectorEvaluator.get_metrics(score_based_evaluation)

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        results = [f'{gamma:.2f}'] + [f'{value * 100:.2f}' for value in results]
        writer.writerow(results)

        f.flush()
