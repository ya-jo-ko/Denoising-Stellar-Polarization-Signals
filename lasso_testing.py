# lasso_testing.py
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


# Perform lasso testing
Original_region_of_interest = data1_test
Mid_SNR_Noisy_region_of_interest = data2_test

D_clean_high_snr = D_h
D_noisy_high_snr = D_l

lasso_lambda = 1e-6 # Sparsity regularization term
#print(np.max(np.abs(np.dot(D_l.T, Mid_SNR_Noisy_region_of_interest)))/D_l.shape[0])
# Suppress ConvergenceWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # Perform Sc_Denoising (with lasso)
    reconstructed_signal_from_Mid_SNR = sc_denoising(Mid_SNR_Noisy_region_of_interest, D_clean_high_snr, D_noisy_high_snr, lasso_lambda)

# Calculate RMSE
Err = (Original_region_of_interest - reconstructed_signal_from_Mid_SNR)**2
rmse1 = np.sqrt(np.mean(Err))

# Print the final RMSE value
print(f"\n\nFinal RMSE: {rmse1}")

# Sample
# Plot original and reconstructed signals
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(Original_region_of_interest[:, -1], 'b')
plt.title('Original Signals')

plt.subplot(1, 3, 2)
plt.plot(Mid_SNR_Noisy_region_of_interest[:, -1], 'g')
plt.title('Noisy Signals')

plt.subplot(1, 3, 3)
plt.plot(reconstructed_signal_from_Mid_SNR[:, -1], 'r')
plt.title('Reconstructed Signals')

plt.show()

# Compute index of change
original_signal = Original_region_of_interest
reconstructed_signal = reconstructed_signal_from_Mid_SNR
rmse_result, change_indices_original, change_indices_reconstructed = compute_rmse_between_change_indices(original_signal, reconstructed_signal)
print("\nRoot Mean Squared Error (RMSE):", rmse_result)
print("Mean Absolute Error:", np.mean(np.abs(change_indices_original-change_indices_reconstructed)))
