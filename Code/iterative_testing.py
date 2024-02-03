# iterative_testing.py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import warnings
import os
import pywt
from sklearn.exceptions import ConvergenceWarning

# Load trained models or any necessary data
output_dir = 'saved_dicts'

D_h = np.load(os.path.join(output_dir, 'D_h.npy'))
D_l = np.load(os.path.join(output_dir, 'D_l.npy'))

# Load split datasets
output_dir = 'saved_data'

data1_train = np.load(os.path.join(output_dir, 'data1_train.npy'))
data1_test = np.load(os.path.join(output_dir, 'data1_test.npy'))
data2_train = np.load(os.path.join(output_dir, 'data2_train.npy'))
data2_test = np.load(os.path.join(output_dir, 'data2_test.npy'))


# Perform iterative testing
# Set the range of alpha and l1_ratio values
alpha_values = np.logspace(-5, 0.6, 8)
l1_ratio_values = np.linspace(0, 1, 10)

# Perform iterative denoising
results = sc_denoising_elastic_net_iterative(Mid_SNR_Noisy_region_of_interest, Original_region_of_interest, D_clean_high_snr, D_noisy_high_snr, alpha_values, l1_ratio_values)

# Find the result with the minimum RMSE
best_result = min(results, key=lambda x: x[2])

# Print the best result
print(f"\nBest Result - Alpha: {best_result[0]}, L1 Ratio: {best_result[1]}, RMSE: {best_result[2]}")

# Access the denoised signal corresponding to the best result
reconstructed_signal_from_Mid_SNR_best = best_result[3]