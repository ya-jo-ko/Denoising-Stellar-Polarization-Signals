# Denoising Stellar Polarization Signals
Assignment for class HY-573, Optimization Methods

## Problem Overview
Stellar polarization signals often exhibit a step function-like model due to changes in polarization as light passes through dust clouds. The goal of this project is to denoise these signals, which are contaminated by Gaussian noise, using coupled dictionary learning and sparse representation techniques.

## Methodology
### Signal Representation
The inherent characteristics of stellar polarization signals are modeled as a step function, introducing sparsity in signal representation. Sparse representation (SR) is employed to express each noisy signal as a sparse linear combination of elements from a trained dictionary.

### Coupled Dictionary Learning
The coupled dictionary learning approach involves creating dictionaries that collectively represent both noisy and clean signal matrices. The objective is to identify paired dictionaries corresponding to the noisy and clean signals. This is achieved by solving a set of sparse decomposition problems with specific constraints.

### Optimization
The optimization process utilizes the Alternating Direction Method of Multipliers (ADMM) technique. It involves updating sparse coefficient matrices, dictionaries, and Lagrange multipliers iteratively. The goal is to reach a stationary point where dictionaries and sparse coefficients provide the best representation of the signals.

### Elastic Net Regularization
An additional Elastic Net regularization approach is implemented to handle both L1 (Lasso) and L2 (Ridge) regularization terms. This provides a balanced solution for variable selection and parameter grouping.

### Detection of Interstellar Magnetized Clouds
To locate the position of interstellar dust clouds, the naive method of maximum change in the signal is used to identify the position of the magnetized cloud.

## How to Run the Code

### 1. Set Up Environment

Make sure you have Python installed on your system. 

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Code

Option 1: Run the Notebook
Open the denoising_sp_signals.ipynb notebook using Jupyter or Google Colab.

Option 2: Run the Training and Testing Scripts
For training dictionaries and testing:

```bash
# Train dictionaries
python train_dicts.py

# Test with Lasso
python test_lasso.py

# Test with Elastic Net
python test_elastic_net.py

# Iterative testing for best parameters with Elastic Net
python iterative_testing_elastic_net.py
```

