# NASA CMAPSS RUL & Risk Prediction

This repository contains a hybrid deep learning model for Remaining Useful Life (RUL) regression and failure **risk** classification on the NASA CMAPSS jet engine simulated dataset (FD001). [file:1]

## Features

- Automatic download of the CMAPSS dataset from Kaggle using `kagglehub`. [file:1]  
- Preprocessing:
  - Column-wise standard deviation analysis and removal of constant features.
  - RUL computation per engine unit from cycle information. [file:1]
  - Sliding-window sequence generation (sequence length = 50 cycles) for time-series modeling. [file:1]
- Multi-task learning:
  - RUL regression head (score_output).
  - Binary high-risk classification head (risk_output) with a configurable RUL threshold (e.g., 30 cycles). [file:1]
- Regularization and training strategies:
  - LSTM-based architecture with dropout, recurrent dropout, and batch normalization.
  - Loss weighting between regression and classification tasks.
  - Early stopping to prevent overfitting.
  - Class weights and per-output sample weights for imbalanced risk labels. [file:1]
- Training curves plotting for MAE and accuracy. [file:1]

## Dataset

The code uses the Kaggle dataset:

- **Name**: `palbha/cmapss-jet-engine-simulated-data`  
- **Source**: https://www.kaggle.com/datasets/palbha/cmapss-jet-engine-simulated-data [file:1][web:2]

The script downloads the dataset automatically using:

```python
import kagglehub
path = kagglehub.dataset_download("palbha/cmapss-jet-engine-simulated-data")
