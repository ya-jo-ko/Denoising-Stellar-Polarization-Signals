# Denoising Stellar Polarization Signals
Assignment for class HY-573, Optimization Methods

## Process Explanation
Stuff to be added.

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

