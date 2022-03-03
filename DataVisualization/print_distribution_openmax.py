import sys
import numpy as np
import pickle
import os
import re
import csv
from tqdm import tqdm
import pandas as pd

from matplotlib import pyplot as plt


def prepare_ax(ax, x_max, x_min, y_max, y_min):
    ax.set_xlim(x_min, x_max)
    ax.set_xticks([])

    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.linspace(y_min, y_max, 11))

# Threshold of 1.45:
#   FPR: 1.00301%
#   FNR: 0.65%

# Threshold of 1.15
#   FPR: 0.10030%
#   FNR: 4.64%

save_by_fault_number = False
fault_metric = 'sdc-1'
threshold = [51.71414 * 0.9,
             5.3407907 * 1.45]

df_list = []
for file_location in tqdm(os.listdir('../FaultyRunResults-OpenMax')):
    seed = re.findall(r'[0-9]+', file_location)[0]
    number_of_faults = re.findall(r'[0-9]+', file_location)[1]
    inference_results = pickle.load(open(f'../FaultyRunResults-OpenMax/{file_location}', 'rb'))

    inference_results['Seed'] = seed
    inference_results['FaultNumber'] = number_of_faults

    df_list.append(inference_results)

faulty_inference_results = pd.concat(df_list, ignore_index=True)
golden_inference_results = pickle.load(open(f'../GoldenRunResults-OpenMax/vgg_imagenet_inference_result.pkl', 'rb'))

if save_by_fault_number:
    unique_number_of_faults = faulty_inference_results.FaultNumber.unique()
    number_of_faults = unique_number_of_faults[3]
    faulty_inference_results = faulty_inference_results[faulty_inference_results.FaultNumber == number_of_faults]

top_golden_values = golden_inference_results.top_1.sort_values(ascending=False).reset_index(drop=True)
top_golden_values = golden_inference_results.iloc[top_golden_values.index.values]
top_golden_values_open_set = top_golden_values[top_golden_values.OpenMaxUnknown]
top_golden_values_under_threshold = top_golden_values.drop(top_golden_values_open_set.index)

top_non_faulty_values = faulty_inference_results[~faulty_inference_results[fault_metric]][['top_1', 'OpenMaxUnknown']].sort_values(by='top_1', ascending=False).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
top_non_faulty_values = faulty_inference_results.iloc[top_non_faulty_values.index.values.tolist()]
top_non_faulty_values_over_threshold = top_non_faulty_values[top_non_faulty_values.OpenMaxUnknown]
top_non_faulty_values_under_threshold = top_non_faulty_values.drop(top_non_faulty_values_over_threshold.index)

top_faulty_values = faulty_inference_results[faulty_inference_results[fault_metric]][['top_1', 'OpenMaxUnknown', 'den']].sort_values(by='top_1', ascending=False).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
top_faulty_values = faulty_inference_results.iloc[top_faulty_values.index.values.tolist()]
top_faulty_values_over_threshold = top_faulty_values[top_faulty_values.OpenMaxUnknown]
top_faulty_values_under_threshold = top_faulty_values.drop(top_faulty_values_over_threshold.index)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

# prepare_ax(ax[0], x_min=0, x_max=len(top_golden_values), y_min=0, y_max=50)
# ax[0].bar(x=top_golden_values_open_set.index, height=top_golden_values_open_set.values, width=1, color='orange', label='False Positive')
# ax[0].bar(x=top_golden_values_under_threshold.index, height=top_golden_values_under_threshold.values, width=1, color='blue', label='True Negative')
# ax[0].legend(loc='best')

prepare_ax(ax[0], x_min=0, x_max=len(top_non_faulty_values), y_min=0, y_max=50)
ax[0].bar(x=top_non_faulty_values_over_threshold.index, height=top_non_faulty_values_over_threshold.values, width=1, color='orange', label='False Positive')
ax[0].bar(x=top_non_faulty_values_under_threshold.index, height=top_non_faulty_values_under_threshold.values, width=1, color='blue', label='True Negative')
ax[0].legend(loc='best')
ax[0].set_xlabel('x - correct predictions')
ax[0].set_ylabel('max(||A(x)||')

# FPR: 0.10030%
print(f'FPR: {100 * len(top_golden_values_open_set) / len(top_golden_values):.5f}%')

prepare_ax(ax[1], x_min=0, x_max=len(top_faulty_values), y_min=0, y_max=50)
ax[1].bar(x=top_faulty_values_over_threshold.index, height=top_faulty_values_over_threshold.values, width=1, color='green', label='True Positive')
ax[1].bar(x=top_faulty_values_under_threshold.index, height=top_faulty_values_under_threshold.values, width=1, color='red', label='False Negative')
ax[1].legend(loc='best')
ax[1].set_xlabel('x - wrong predictions')
ax[1].set_ylabel('max(||A(x)||')

# FNR: 4.64%
print(f'FNR: {100 * len(top_faulty_values_under_threshold) / len(top_faulty_values):.2f}%')

# fig.show()
if save_by_fault_number:
    os.makedirs(f'ByFaultNumbers/{fault_metric}/{number_of_faults.zfill(3)}')
    fig.savefig(f'ByFaultNumbers/{fault_metric}/{number_of_faults.zfill(3)}/Distribution-no_fault_detector-faults-1')
else:
    os.makedirs(f'{fault_metric}', exist_ok=True)
    fig.savefig(f'{fault_metric}/Distribution-no_fault_detector-faults-1')
