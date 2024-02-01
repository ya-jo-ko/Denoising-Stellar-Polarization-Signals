import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pywt
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import warnings as warnings
from sklearn.exceptions import ConvergenceWarning

def download_data(choose_folder, choose_file):
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
    folders = {
        'clean': {
            'nn_input_u_cs_39dcs': '1qlIDtwK0ONDXrjByFY8rDiu5oqxCa0st',
            #'nn_output_cs_39dcs': '1qlIDtwK0ONDXrjByFY8rDiu5oqxCa0st',
            'nn_input_q_cs_39dcs': '1cWMYPytdknj_QDyJDQuEWXtWXaYyqq79',
        },
        'low_noise': {
            'nn_input_u_cs_39dcs': '1RRRzAkU_nQbXdUQgYP3OUsNxXAeU3ue_',
            #'nn_output_cs_39dcs': '1CcD1Mo0yeIru7I85KIowfCaxaPZ_w1-m',
            'nn_input_q_cs_39dcs': '1tB3XEOGhkAsQRZnn95mQ-nfk8J-JCjvx',
        },
        'more_noise': {
            'nn_input_u_cs_39dcs': '1wgf45RlC6Dpf10DKg8Ec1PlmoScU8yVw',
            #'nn_output_cs_39dcs': '1LhTztFP5L9NCM4I8qiEZGOoO6ytNMb3U',
            'nn_input_q_cs_39dcs': '1NCc_Q-nl6-4TfmBADkTLIlK5TP8NKi6D',
        },
        'evenmore_noise': {
            'nn_input_u_cs_39dcs': '19IZuoG30-1QAUqwtqC7aJcsnhYNxVEvB',
            #'nn_output_cs_39dcs': '1BtAyHVQSRyTInXNn25O8JL8CpgvINdnQ',
            'nn_input_q_cs_39dcs': '1aqOSDVcWY-07k6RzHvdJaUXJIZ3E8wdQ',
        },
    }

    if choose_folder not in folders:
        print('Directory name not found...')
        return

    folder_data = folders[choose_folder]
    if choose_file not in folder_data:
        print('File name not found...')
        return

    file_id = folder_data[choose_file]

    # Construct the download link
    download_link = f'https://drive.google.com/uc?id={file_id}'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(download_link)
    return df

def plot_signal(time, signal, average_over=5, figname=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    #ax.plot(time, signal[:, 0], label='q')
    #ax.plot(time, signal[:, 1], label='u')
    ax.plot(time, signal, label='q')
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Stokes Parameters', fontsize=18)
    ax.set_xlabel('Distance (pc)', fontsize=18)
    ax.legend()
    plt.show()

def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = np.fft.fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values

def plot_fft_plus_power(time, signal, figname=None):
    dt = time[1] - time[0]
    N = len(signal)
    fs = 1/dt
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    variance = np.std(signal)**2
    f_values, fft_values = get_fft_values(signal, dt, N, fs)
    fft_power = variance * abs(fft_values) ** 2  # FFT power spectrum
    ax[0].plot(f_values, fft_values, 'r-', label='Fourier Transform')
    ax[1].plot(f_values, fft_power, 'k--',
               linewidth=1, label='FFT Power Spectrum')
    ax[1].set_xlabel('Frequency [Hz / year]', fontsize=18)
    ax[1].set_ylabel('Amplitude', fontsize=12)
    ax[0].set_ylabel('Amplitude', fontsize=12)
    ax[0].legend()
    ax[1].legend()
    plt.show()


def soft_thresh(b, lambda_val):
    # Set the threshold
    th = lambda_val / 2

    # First find elements that are larger than the threshold
    x = np.zeros_like(b)
    k = np.where(b > th)[0]
    x[k] = b[k] - th

    # Next find elements that are less than abs
    k = np.where(np.abs(b) <= th)[0]
    x[k] = 0

    # Finally find elements that are less than -th
    k = np.where(b < -th)[0]
    x[k] = b[k] + th

    return x

def normalize_matrix(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = matrix * 1/np.sqrt(np.sum(matrix * matrix))
    return normalized_matrix


def normalize_data(matrix):
    normalized_matrix = []
    for i in range(matrix.shape[1]):
      min_val = np.min(matrix[:, i])
      max_val = np.max(matrix[:, i])
      normalized_matrix_i = (matrix[:, i]- min_val) / (max_val - min_val)
      normalized_matrix.append(normalized_matrix_i)
    return np.array(normalized_matrix).T

def normalize_dict(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = 2 * (matrix - min_val) / (max_val - min_val) -1
    return normalized_matrix


def ADMM_Coupled_DL(params):
    # Parse input parameters
    data1 = params['data1']
    data2 = params['data2']
    dictsize = params['dictsize']
    iternum = params['iternum']

    # Initialize the dictionary
    idx = np.random.permutation(data1.shape[1])
    data1 = data1[:, idx]
    data2 = data2[:, idx]

    D_h = data1[:, :dictsize]
    D_l = data2[:, :dictsize]

    # Normalize the dictionary
    print(f"Before normalization: D_h.shape = {D_h.shape}")
    #D_h = np.dot(D_h,np.diag(1.0 / np.sqrt(np.sum(D_h * D_h, axis=0))))
    D_h = normalize_dict(D_h)
    print(f"After normalization: D_h.shape = {D_h.shape}")
    #D_l = np.dot(D_l,np.diag(1.0 / np.sqrt(np.sum(D_l * D_l, axis=0))))
    D_l = normalize_dict(D_l)

    c1 = 0.4
    c2 = 0.4
    c3 = 0.8
    maxbeta = 1e6  # Set the maximum mu

    delta = 1e-4
    beta = 0.01

    W_h = np.random.randn(params['dictsize'], data1.shape[1])
    W_l = np.random.randn(params['dictsize'], data2.shape[1])

    Y1 = np.zeros_like(W_h)
    Y2 = np.zeros_like(W_h)
    Y3 = np.zeros_like(W_h)

    I = np.eye(dictsize)

    P = W_h.copy()
    Q = W_l.copy()

    err1 = np.zeros(iternum)
    err2 = np.zeros(iternum)

    # Main loop
    for iter in range(iternum):
        W_h = np.dot(np.linalg.inv(np.dot(D_h.T,D_h) + c1 * I + c3* I), (np.dot(D_h.T, data1) + Y1-Y3+c1*P + c3* W_l))
        W_l = np.dot(np.linalg.inv(np.dot(D_l.T,D_l) + c2 * I + c3* I), (np.dot(D_l.T, data2) + Y2-Y3+c2*Q + c3* W_h))

        # Update P and Q
        for xx in range(data1.shape[1]):
            y_1 = (W_h[:, xx] - Y1[:, xx] / c1)
            y_1 = y_1 / np.linalg.norm(y_1)
            #tmp_1 = np.clip(y_1, -0.2, 0.2)
            tmp_1 = soft_thresh(y_1,0.2)
            P[:, xx] = tmp_1

            y_2 = (W_l[:, xx] - Y2[:, xx] / c2)
            y_2 = y_2 / np.linalg.norm(y_2)
            #tmp_2 = np.clip(y_2, -0.2, 0.2)
            tmp_2 = soft_thresh(y_2,0.2)
            Q[:, xx] = tmp_2

        # Update D_h and D_l
        for j in range(dictsize):
            phi_1 = np.dot(W_h[j, :], W_h[j, :].T)
            phi_2 = np.dot(W_l[j, :], W_l[j, :].T)
            D_h[:, j] = D_h[:, j] + np.dot(data1, W_h[j, :].T) / (phi_1 + delta)
            D_l[:, j] = D_l[:, j] + np.dot(data2, W_l[j, :].T) / (phi_2 + delta)

        # Normalize D_h and D_l
        #D_h = np.dot(D_h,np.diag(1.0 / np.sqrt(np.sum(D_h * D_h, axis=0))))
        #D_l = np.dot(D_l,np.diag(1.0 / np.sqrt(np.sum(D_l * D_l, axis=0))))
        D_h = normalize_dict(D_h)
        D_l = normalize_dict(D_l)

        # Update Lagrange multipliers
        Y1 = Y1 + c1 * np.minimum(maxbeta, beta * c1) * (P - W_h)
        Y2 = Y2 + c2 * np.minimum(maxbeta, beta * c2) * (Q - W_l)
        Y3 = Y3 + c3 * np.minimum(maxbeta, beta * c3) * (W_h - W_l)

        # Calculate errors
        err1[iter] = np.sqrt(np.sum((data1 - np.dot(D_h, W_h))**2) / np.prod(data1.shape))
        err2[iter] = np.sqrt(np.sum((data2 - np.dot(D_l, W_l))**2) / np.prod(data2.shape))

        info = f'Iteration {iter + 1} / {iternum} complete, RMSE1 = {err1[iter]}, RMSE2 = {err2[iter]}'
        print(info)

    return D_h, D_l, P, Q, err1, err2


def solve_lasso(D, y, alpha):
    # Solves the Lasso problem using scikit-learn's Lasso
    model = Lasso(alpha=alpha, fit_intercept=False)#false
    model.fit(D, y)
    return model.coef_

def solve_linear_regression(D, y):
    # Solves the linear regression problem using scikit-learn's LinearRegression
    model = LinearRegression(fit_intercept=False)
    model.fit(D, y)
    return model.coef_

def sc_denoising(noisy_in, Dh, Dl, lambda_val):
    # Initialization
    denoised = np.zeros_like(noisy_in)

    for jj in tqdm(range(noisy_in.shape[1]), desc="Denoising Progress"):
        # print(jj / noisy_in.shape[1])
        noisy_part = noisy_in[:, jj]
        # Calculate the norm
        mNorm = np.sqrt(np.sum(noisy_part ** 2))
        #print(mNorm)
        # Normalize the input noisy signal
        if mNorm != 0:
            y = noisy_part / mNorm
        else:
            y = noisy_part
        #y = noisy_part
        # Solve the sparse decomposition problem using Lasso
        w = solve_lasso(Dl, y, lambda_val)
        # Solve the sparse decomposition problem using Linear Regression
        #w = solve_linear_regression(Dl, y)
        #w = np.dot(np.linalg.inv(np.dot(D_l,D_l.T)),y)

        # Map to the clean high-resolution Dictionary
        denoised_sig = np.dot(Dh,w) * mNorm
        denoised[:, jj] = denoised_sig

    return denoised

def compute_rmse_between_change_indices(original_signal, reconstructed_signal):
  reconstructed_signal_normalized = reconstructed_signal#normalize_matrix(reconstructed_signal)
  change_indices_original = []
  mean_change_indices_reconstructed = []
  for jj in range(original_signal.shape[1]):
    # Find indices where values change in the original signal
    change_indices_original_column = np.where(np.diff(original_signal[:, jj]) != 0)[0] + 1
    if not len(change_indices_original_column>0):
      x=0
    else:
      first_change_index_original = change_indices_original_column[0] if len(change_indices_original_column) > 0 else 0
      change_indices_original.append(first_change_index_original)

      # Find indices where values change in the reconstructed signal
      change_indices_reconstructed = np.where(np.diff(reconstructed_signal_normalized[:,jj]) != 0)[0] + 1
      min = np.min(change_indices_reconstructed)
      maxim = np.max(change_indices_reconstructed)
      #median = (np.max(change_indices_reconstructed)-np.min(change_indices_reconstructed))/2
      # Compute the mean of change indices in the reconstructed signal
      #mean_change_indices_reconstructed.append(np.mean(change_indices_reconstructed))
      # Calculate a derivative-like value for the reconstructed signal
      derivative_like_values = np.abs(np.diff(reconstructed_signal_normalized[min:maxim, jj]))

      # Find the index with the highest derivative-like value
      index_highest_derivative = np.argmax(derivative_like_values) + min + 1
      # Add the index with the highest derivative-like value to the mean_change_indices_reconstructed list
      mean_change_indices_reconstructed.append(index_highest_derivative)

  # Compute the root mean squared error (RMSE) between original and mean reconstructed change indices
  print('Original Signal Change Index:')
  print(change_indices_original)
  print('Reconstructed Signal Change Index:')
  print(mean_change_indices_reconstructed)
  change_indices_original = np.array(change_indices_original)
  mean_change_indices_reconstructed = np.array(mean_change_indices_reconstructed)
  # normalize indexes to calculate error
  #change_indices_original_normalized = change_indices_original/original_signal.shape[0]
  #mean_change_indices_reconstructed_normalized = mean_change_indices_reconstructed/reconstructed_signal.shape[0]
  rmse = np.sqrt(np.mean((change_indices_original - mean_change_indices_reconstructed) ** 2))

  return rmse,change_indices_original,mean_change_indices_reconstructed

def plot_change_indices(change_indices_original, mean_change_indices_reconstructed):
    plt.figure(figsize=(8, 5))

    # Plot change indices
    plt.plot(change_indices_original, color='blue', label='Original Change Indices')
    plt.plot(mean_change_indices_reconstructed, color='green', label='Reconstructed Change Indices')

    plt.ylabel('Change Indices')
    plt.title('Change Indices Comparison')
    plt.legend()

    plt.show()



def solve_elastic_net(D, y, alpha, l1_ratio):
    # Solves the Elastic Net problem using scikit-learn's ElasticNet
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False)
    model.fit(D, y)
    return model.coef_

def sc_denoising_elastic_net(noisy_in, Dh, Dl, alpha, l1_ratio):
    # Initialization
    denoised = np.zeros_like(noisy_in)

    for jj in tqdm(range(noisy_in.shape[1]), desc="Denoising Progress"):
        # print(jj / noisy_in.shape[1])
        noisy_part = noisy_in[:, jj]
        # Calculate the norm
        mNorm = np.sqrt(np.sum(noisy_part ** 2))
        # Normalize the input noisy signal
        if mNorm != 0:
            y = noisy_part / mNorm
        else:
            y = noisy_part

        # Solve the sparse decomposition problem using Elastic Net
        w = solve_elastic_net(Dl, y, alpha, l1_ratio)

        # Map to the clean high-resolution Dictionary
        denoised_sig = np.dot(Dh, w) * mNorm
        denoised[:, jj] = denoised_sig

    return denoised

def sc_denoising_elastic_net_iterative(noisy_in, Original_region_of_interest, Dh, Dl, alpha_values, l1_ratio_values):
    results = []

    for alpha in alpha_values:
        for l1_ratio in l1_ratio_values:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                # Denoise using Elastic Net with current alpha and l1_ratio
                reconstructed_signal = sc_denoising_elastic_net(noisy_in, Dh, Dl, alpha, l1_ratio)

                # Calculate RMSE
                Err = (Original_region_of_interest - reconstructed_signal)**2
                rmse = np.sqrt(np.mean(Err))

                results.append((alpha, l1_ratio, rmse, reconstructed_signal))

    return results

