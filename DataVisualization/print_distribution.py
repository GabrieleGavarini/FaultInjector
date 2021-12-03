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


threshold = [51.71414 * 0.9,
             5.3407907 * 1.3]

df_list = []
for file_location in tqdm(os.listdir('../FaultyRunResults')):
    seed = re.findall(r'[0-9]+', file_location)[0]
    number_of_faults = re.findall(r'[0-9]+', file_location)[1]
    inference_results = pickle.load(open(f'../FaultyRunResults/{file_location}', 'rb'))

    inference_results['Seed'] = seed
    inference_results['FaultNumber'] = number_of_faults

    df_list.append(inference_results)

faulty_inference_results = pd.concat(df_list, ignore_index=True)
golden_inference_results = pickle.load(open(f'../GoldenRunResults/vgg_imagenet_inference_result.pkl', 'rb'))

top_golden_values = golden_inference_results.top_1.sort_values(ascending=False).reset_index(drop=True)
top_golden_values_over_threshold = top_golden_values[(top_golden_values.values > threshold[0]) | (top_golden_values.values < threshold[1])]
top_golden_values_under_threshold = top_golden_values.drop(top_golden_values_over_threshold.index)

top_non_faulty_values = faulty_inference_results[~faulty_inference_results['sdc-1']].top_1.sort_values(ascending=False).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
top_non_faulty_values_over_threshold = top_non_faulty_values[(top_non_faulty_values.values > threshold[0]) | (top_non_faulty_values.values < threshold[1])]
top_non_faulty_values_under_threshold = top_non_faulty_values.drop(top_non_faulty_values_over_threshold.index)

top_faulty_values = faulty_inference_results[faulty_inference_results['sdc-1']].top_1.sort_values(ascending=False).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
top_faulty_values_over_threshold = top_faulty_values[(top_faulty_values.values > threshold[0]) | (top_faulty_values.values < threshold[1])]
top_faulty_values_under_threshold = top_faulty_values.drop(top_faulty_values_over_threshold.index)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 3))

# prepare_ax(ax[0], x_min=0, x_max=len(top_golden_values), y_min=0, y_max=50)
# ax[0].bar(x=top_golden_values_over_threshold.index, height=top_golden_values_over_threshold.values, width=1, color='orange', label='False Positive')
# ax[0].bar(x=top_golden_values_under_threshold.index, height=top_golden_values_under_threshold.values, width=1, color='blue', label='True Negative')
# ax[0].legend(loc='best')

prepare_ax(ax[0], x_min=0, x_max=len(top_non_faulty_values), y_min=0, y_max=50)
ax[0].bar(x=top_non_faulty_values_over_threshold.index, height=top_non_faulty_values_over_threshold.values, width=1, color='orange', label='False Positive')
ax[0].bar(x=top_non_faulty_values_under_threshold.index, height=top_non_faulty_values_under_threshold.values, width=1, color='blue', label='True Negative')
ax[0].legend(loc='best')

print(f'FPR: {100 * len(top_golden_values_over_threshold) / len(top_golden_values):.5f}%')

prepare_ax(ax[1], x_min=0, x_max=len(top_faulty_values), y_min=0, y_max=50)
ax[1].bar(x=top_faulty_values_over_threshold.index, height=top_faulty_values_over_threshold.values, width=1, color='green', label='True Positive')
ax[1].bar(x=top_faulty_values_under_threshold.index, height=top_faulty_values_under_threshold.values, width=1, color='red', label='False Negative')
ax[1].legend(loc='best')

print(f'FNR: {100 * len(top_faulty_values_under_threshold) / len(top_faulty_values):.2f}%')

fig.show()
