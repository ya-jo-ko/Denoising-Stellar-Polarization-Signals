# train_dicts.py
import numpy as np
from utils import *
import matplotlib.pyplot as plt
import warnings
import os
import pywt
from sklearn.model_selection import train_test_split 
from sklearn.exceptions import ConvergenceWarning

# Download dataset and perform training
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Available folder options:
  - 'clean'
  - 'low_noise'
  - 'more_noise'
  - 'evenmore_noise'

Available file options:
  - 'nn_input_q_cs_39dcs'
  - 'nn_input_u_cs_39dcs'
  - 'nn_output_q_cs_39dcs'
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''

# Example usage for downloading a dataset
choose_folder = 'more_noise'
choose_file = 'nn_input_u_cs_39dcs'
choose_folder_ = 'more_noise'
choose_file_ = 'nn_input_q_cs_39dcs'

# ~~~~~~~~~~ CHOOSE FOLDER 2 SHOULD ALWAYS BE CLEAN ~~~~~~~~~~
choose_folder_2 = 'clean'
choose_file_2 = 'nn_input_u_cs_39dcs'
choose_folder_2_ = 'clean'
choose_file_2_ = 'nn_input_q_cs_39dcs'

df_11 = download_data(choose_folder_2, choose_file_2)
df_12 = download_data(choose_folder_2_, choose_file_2_)
df_21 = download_data(choose_folder, choose_file)
df_22 = download_data(choose_folder_, choose_file_)

# Stack DataFrames as arrays
array_1 = np.vstack([df_11.values, df_12.values])
array_2 = np.vstack([df_21.values, df_22.values])

# Combine arrays vertically
combined_array = np.vstack([array_1, array_2])

# Create a new DataFrame from the combined array
df_1 = pd.DataFrame(array_1)
df_2 = pd.DataFrame(array_2)

# Display the combined DataFrame
print("Combined DataFrame 1 shape:", df_1.shape)
print("Combined DataFrame 2 shape:", df_2.shape)

# Train test split
# Extract relevant columns from DataFrames and convert to NumPy arrays
data1 = df_1.copy().values
data2 = df_2.copy().values

# Set the random seed for reproducibility
random_seed = 42

# Split the datasets into train and test sets
data1_train, data1_test = train_test_split(data1, test_size=0.2, random_state=random_seed)
data2_train, data2_test = train_test_split(data2, test_size=0.2, random_state=random_seed)

# Transpose
data1_train = np.transpose(data1_train)
data1_test = np.transpose(data1_test)
data2_train = np.transpose(data2_train)
data2_test = np.transpose(data2_test)

# Normalize
data1_train = normalize_dict(data1_train)
data1_test = normalize_dict(data1_test)
data2_train = normalize_dict(data2_train)
data2_test = normalize_dict(data2_test)

# Display the shapes of the resulting sets
print("data1_train shape:", data1_train.shape)
print("data1_test shape:", data1_test.shape)
print("data2_train shape:", data2_train.shape)
print("data2_test shape:", data2_test.shape)

# Training
# Set parameters
dictsize = data1_train.shape[1]
iternum = 50

# Set parameters for ADMM_Coupled_DL function
params = {
    'data1': data1_train,
    'data2': data2_train,
    'dictsize': dictsize,
    'iternum': iternum
}

print(['Dictsize: ',{dictsize}])

# Call the ADMM_Coupled_DL function
D_h, D_l, P, Q, err1, err2 = ADMM_Coupled_DL(params)

# RMSE_1 is for noisy, RMSE_2 is for clean

# Plot the errors over iterations
plt.plot(range(iternum), err1, label='RMSE1')
plt.plot(range(iternum), err2, label='RMSE2')
plt.xlabel('Iteration')
plt.ylabel('RMSE')
plt.legend()
plt.title('Convergence of RMSE')
plt.show()

# Save the split datasets and trained models locally
output_dir = 'saved_data'
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

# Save split datasets
np.save(os.path.join(output_dir, 'data1_train.npy'), data1_train)
np.save(os.path.join(output_dir, 'data1_test.npy'), data1_test)
np.save(os.path.join(output_dir, 'data2_train.npy'), data2_train)
np.save(os.path.join(output_dir, 'data2_test.npy'), data2_test)


# Save dictionaries 
output_dir = 'saved_dicts'
os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist

np.save(os.path.join(output_dir, 'D_h.npy'), D_h)
np.save(os.path.join(output_dir, 'D_l.npy'), D_l)
