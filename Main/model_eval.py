# Load the trained model of choice

# Import statements
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns
# from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
import torchmetrics
from enum import Enum
import csv
from sklearn.utils import shuffle

ACTIVITY_FRAMES = 150 # Number of frames for a single activity sample (6s).

# Image dimensions are based on the radar configuration, these need to be set and changed inside of this file accordingly.
X_DIM = 11 #36 # For example.
Y_DIM = 7 #18 # For example.
XY_DIM = X_DIM * Y_DIM

PATH = 'C:\\Users\\Samuel Mason\\Documents\\GitHub\\mmWave-HAR\\Main\\CNN-LSTM_model.pth'

model = torch.load(PATH)
print("Model loaded successfully!")
model.eval()