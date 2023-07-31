# CNN-LSTM Model for testing on gathered human activity data.

# Using pytorch implementation.

# Import statements
import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylabimport rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim

# Matplotlib additional arguments.
%matplotlib inline
%config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

activity_data = pd.read_csv('activity_data.csv') # We are assuming that the data is in csv file format here, if it isn't, then we need to do some additional pre-processing.

# Assume the data is in the format:
# Activity sample 1: Image 1, Image 2, Image 3, etc. 
# Activity sample 2: Image 1, Image 2, Image 3, etc.
# etc.

# Then, reading a single row will give an entire activity, we need to split this activity according to the size of the captured frames.

# If a captured frame has, for example, a resolution of 100x100 px, then we can take the first 10000 values as the first image, and reshape accordingly.
# Although the heatmap displays two different colour extremes (blue and orange), the image data is still 2-dimensional.

# Start by getting the number of lines in the csv, this is how many activities we have data for.

reader = csv.reader(open('activity_data.csv'))
activity_count = len(list(reader))

# Image dimensions are based on the radar configuration, these need to be set and changed inside of this file accordingly.
X_DIM = 80 # For example.
Y_DIM = 100 # For example.
XY_DIM = X_DIM * Y_DIM

def format_sequences(data, count):

    px = [] # This list will contain a row of pixels for a single frame.
    py = [] # This list will contain a set of pixel rows for a single frame (one frame).
    pt = [] # This list will contain the sequence of frames.

    xptr = 0 # Pointer for the end of the most recent row.
    yptr = 0 # Pointer for the end of the most recent frame.

    for i in range(count):
        this_activity = activity_data.iloc[i] # Get the next activity.

        # This will continue for the required number of frame iterations.
        for j in range(len(this_activity) / XY_DIM):
            for k in range(Y_DIM):
                px = this_activity.iloc[xptr * X_DIM + yptr * XY_DIM: (xptr + 1) * X_DIM + yptr * XY_DIM]
                py.append(px)
                xptr += 1
            pt.append(py)
            yptr += 1
            xptr = 0

    # At the end of this code execution, pt will contain lists of lists, representing the 2D images.

