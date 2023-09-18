# CNN-LSTM Model for testing on gathered human activity data.
# Using pytorch implementation.

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
from enum import Enum
import csv

# Class names will change based on the information being fed to the network; this is just an example.
class ClassNames(Enum):
    JUMPING = 0
    WALKING = 1
    CLAPPING = 2

# Matplotlib additional arguments (jupyter notebook only).
# %matplotlib inline
# %config InlineBackend.figure_format='retina'

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
# rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

path = 'C:\\Data\\output.csv' # Path to the csv file with all of the activity data.
print(f"Path being used is: {path}")

# Read in the activity data, create a dataframe for it with the rows and columns transposed for optimisation.
activity_data = open(path).readlines()
del activity_data[0]

# Total number of activities recorded.
activity_count = len(activity_data)

df = pd.DataFrame()

for i in range(activity_count):
    temp = activity_data[i].split(',')
    temp[-1] = temp[-1].split('\n')[0] # Remove the newline character at the end.
    activity_name = temp[0] 
    temp = temp[1:] # Remove the activity name (it is a string, not complex dtype, so separate it).
    cmplx = []
    for s in temp:
        s = s.replace('i', 'j')
        v = abs(complex(s)) # Use the modulus of the complex number, else stuff gets wacky (type errors).
        cmplx.append(v)
    cmplx.insert(0, activity_name)
    df[i] = cmplx

# Print as a test.
print(df[0])

# activity_data = pd.read_csv(path) # We are assuming that the data is in csv file format here, if it isn't, then we need to do some additional pre-processing.
print("Activity data has been read in")

# Assume the data is in the format:
# Activity sample 1: Class, Image 1, Image 2, Image 3, etc. 
# Activity sample 2: Class, Image 1, Image 2, Image 3, etc.
# etc.

# Then, reading a single row will give an entire activity, we need to split this activity according to the size of the captured frames.

# If a captured frame has, for example, a resolution of 100x100 px, then we can take the first 10000 values as the first image, and reshape accordingly.
# Although the heatmap displays two different colour extremes (blue and orange), the image data is still 2-dimensional.

# Start by getting the number of lines in the csv, this is how many activities we have data for.

print(f"Activity count is {activity_count}")

# Image dimensions are based on the radar configuration, these need to be set and changed inside of this file accordingly.
X_DIM = 50 # For example.
Y_DIM = 100 # For example.
XY_DIM = X_DIM * Y_DIM

def format_sequences(df, count):
    px = [] # This list will contain a row of pixels for a single frame.
    py = [] # This list will contain a set of pixel rows for a single frame (one frame).
    pt = [] # This list will contain the sequence of frames.

    classes = [] # This list will contain the classes for each of the activity sequences.
    class_nums = [] # This list con

    xptr = 0 # Pointer for the end of the most recent row.
    yptr = 0 # Pointer for the end of the most recent frame.

    df = df.T # Transpose the data frame.

    # We need to start at 1 because the first values are not reliable.
    for i in range(0, count - 1):
        this_activity = df.iloc[i] # Get the next activity.
        classes.append(this_activity.iloc[0]) # Append the class.
        this_activity = this_activity.iloc[1:] # Remove the class before formatting the rest of the data.
        this_activity = this_activity.iloc[:500000]
        print(len(this_activity))
        print(this_activity)
        #print(this_activity)
        # This will continue for the required number of frame iterations.
        for j in range(round(len(this_activity) / XY_DIM)):
            for k in range(Y_DIM):
                px = this_activity.iloc[xptr * X_DIM + yptr * XY_DIM: (xptr + 1) * X_DIM + yptr * XY_DIM]
                py.append(px)
                xptr += 1
            pt.append(py)
            yptr += 1
            xptr = 0

    # Now we need to convert the classes to a numerical (enumerated) representation.
    for i in range(len(classes)):
        this_class = classes[i].split('_')[1]
        # Can replace this with a better structure if needed, for 3 classes this should be sufficient for now.
        if this_class == "jump":
            class_nums.append(ClassNames.JUMPING)
        elif this_class == "walk":
            class_nums.append(ClassNames.WALKING)
        elif this_class == "clap":
            class_nums.append(ClassNames.CLAPPING)

    # At the end of this code execution, pt will contain lists of lists, representing the 2D images.
    # Class nums contains a numerical representation of the classes, which can be used for training purposes.
    return pt, class_nums

# Format the data from csv.
X, y = format_sequences(df, activity_count)

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

class CNNLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(CNNLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        # 2D CNN layer.
        self.c = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 3)
        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers
        )
        self.linear = nn.Linear(in_features = n_hidden, out_features = 1)

    # IMPORTANT! This assumes that the sequences input from the csv are of a uniform length - if for some reason they are not, you need to add additional code to make sure that they are the same length.
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len - 1, self.n_hidden),
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

def train_model(model, train_data, train_labels, val_data = None, val_labels = None, num_epochs = 100, verbose = 10, patience = 10):
    loss_fn = torch.nn.L1Loss() # L1 loss by default.
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.001) # Default learning rate is 0.001.
    # Histograms used to monitor training progress.
    train_hist = []
    val_hist = []

    for t in range(num_epochs):
        epoch_loss = 0

        for idx, seq in enumerate(train_data):

            # After every sample, we need to reset the hidden state.
            model.reset_hidden_state()

            seq = torch.unsqueeze(seq, 0)
            y_pred = model(seq)
            loss = loss_fn(y_pred[0].float(), train_labels[idx]) # Calculate the loss after 1 step, then update the weights.

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        train_hist.append(epoch_loss / len(train_data))

        if val_data is not None:

            with torch.no_grad():

                val_loss = 0

                for val_idx, val_seq in enumerate(val_data):

                    model.reset_hidden_state() # Reset the hidden state with every sequence.

                    val_seq = torch.unsqueeze(val_seq, 0)
                    y_val_pred = model(val_seq)
                    val_step_loss = loss_fun(y_val_pred[0].float(), val_labels[val_idx])

                    val_loss += val_step_loss

            val_hist_append(val_loss / len(val_data))

            # Print the loss based on verbose value (pseudo-verbose).
            if t % verbose == 0:
                print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')

            # Can add early stopping if wanted, not using currently.

    return model, train_hist, val_hist

seq_length = 500 # This needs to be tailored - based on the number of frames in a captured sequence of activity data.

model = CNNLSTM(
    n_features = 1,
    n_hidden = 4,
    seq_len = seq_length,
    n_layers = 1
)

model, train_hist, val_hist = train_model(
    model, 
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs = 100,
    verbose = 10,
    patience = 50
)

plt.figure(figsize = (14, 10)) # Attempt to set the figure size using an alternative method.
plt.plot(train_hist, label = "Training loss")
plt.plot(val_hist, label = "Val loss")
plt.legend()



