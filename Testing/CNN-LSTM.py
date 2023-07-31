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
# Activity sample 1: Class, Image 1, Image 2, Image 3, etc. 
# Activity sample 2: Class, Image 1, Image 2, Image 3, etc.
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

    classes = [] # This list will contain the classes for each of the activity sequences.

    xptr = 0 # Pointer for the end of the most recent row.
    yptr = 0 # Pointer for the end of the most recent frame.

    for i in range(count):
        this_activity = activity_data.iloc[i] # Get the next activity.
        classes.append(this_activity.iloc[0]) # Append the class.
        this_activity = this_activity.iloc[1:] # Remove the class before formatting the rest of the data.

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
    return pt, classes

# Format the data from csv.
X, y = format_sequences(activity_data, activity_count)

scaler = MinMaxScaler()
X = scaler.fit_transform(X) # Normalise the input data.

# Divide the dataset into training, validation and testing sets.
train_size = int(activity_count * 0.8)
val_size = int(activity_count * 0.1)
# Testing size is just what is leftover.

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Transform to a tensor.
def make_Tensor(array):
	return torch.from_numpy(array).float()

X_train = make_Tensor(X_train)
y_train = make_Tensor(y_train)
X_val = make_Tensor(X_val)
y_val = make_Tensor(y_val)
X_test = make_Tensor(X_test)
y_test = make_Tensor(y_test)

class CNN-LSTM(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(CNN-LSTM, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        # 2D CNN layer.
        self.c = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3)
        self.lstm = nn.LSTM(
            input_size = n_features
            hidden_size = n_hidden
            num_layers = n_layers
        )
        self.linear = nn.Linear(in_features = n_hidden, out_features = 1)

    # IMPORTANT! This assumes that the sequences input from the csv are of a uniform length - if for some reason they are not, you need to add additional code to make sure that they are the same length.
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len - 1, self.n_hidden)
            torch.zeros(self.n_layers, self.seq_len - 1, self.n_hidden)
        )

    def forward(self, seq):
        seq = self.c(seq.view(len(seq), 1, -1))
        lstm_out, self.hidden = self.lstm(
            seq.view(len(seq), self.seq_len - 1, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len - 1, len(seq), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred





