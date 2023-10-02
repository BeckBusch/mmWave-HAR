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

# TODO:

# Defines
INPUT_MIN_FRAMES = 8 # placeholder
INPUT_MAX_FRAMES = 64 # placeholder

# From planning:
# Input size (min e.g. 8 to max e.g. 64)
# (Generator) Conform to fixed size e.g. 8 using analogue to PCA
# Match first N-k points e.g. 7 (Classifier)
# After the terminal point is observed, start a new observation window

# Continuous input stream size is variable from starting point up to maximum, which is defined.

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

print(f"Activity count is {activity_count}\n")

# Image dimensions are based on the radar configuration, these need to be set and changed inside of this file accordingly.
X_DIM = 50 # For example.
Y_DIM = 100 # For example.
XY_DIM = X_DIM * Y_DIM

def format_sequences(df, count):
    px = [] # This list will contain a row of pixels for a single frame.
    py = [] # This list will contain a set of pixel rows for a single frame (one frame).
    pt = [] # This list will contain the sequence of frames.

    pT = [] # This list will contain everything (one item per class label).

    pp = np.zeros(X_DIM) # This is a compression array, used for storing compressed pixel values.
    vert_comp_factor = 10 # Vertical compression factor (y values).

    classes = [] # This list will contain the classes for each of the activity sequences.
    class_nums = [] # This list contains the enumerated classes.

    xptr = 0 # Pointer for the end of the most recent row.
    yptr = 0 # Pointer for the end of the most recent frame.

    df = df.T # Transpose the data frame.

    # We need to start at 1 because the first values are not reliable.
    for i in range(count):
        this_activity = df.iloc[i] # Get the next activity.
        classes.append(this_activity.iloc[0]) # Append the class.
        this_activity = this_activity.iloc[1:] # Remove the class before formatting the rest of the data.
        this_activity = this_activity.iloc[:500000]
        print(i)
        #print(this_activity)
        # This will continue for the required number of frame iterations.
        for j in range(round(len(this_activity) / XY_DIM)):
            for k in range(Y_DIM):
                px = this_activity.iloc[xptr * X_DIM + yptr * XY_DIM: (xptr + 1) * X_DIM + yptr * XY_DIM]
                if (len(px) != X_DIM) :
                    px = np.zeros(X_DIM)
                pp = np.add(pp, px)
                if (k % vert_comp_factor == 0):
                    pp = np.divide(pp, vert_comp_factor)
                    py.append(pp)
                    pp = np.zeros(X_DIM)
                xptr += 1
            pt.append(py)
            py = [] # Reset py.
            yptr += 1
            xptr = 0
        pp = np.zeros(X_DIM) # Reset after each inner loop.
        pT.append(pt)
        pt = [] # Reset pt.

    # Now we need to convert the classes to a numerical (enumerated) representation.
    for i in range(len(classes)):
        this_class = classes[i].split('_')[1]
        # Can replace this with a better structure if needed, for 3 classes this should be sufficient for now.
        if this_class == "jump":
            class_nums.append(ClassNames.JUMPING)
        elif this_class == "jumps":
            class_nums.append(ClassNames.JUMPING)
        elif this_class == "walk":
            class_nums.append(ClassNames.WALKING)
        elif this_class == "clap":
            class_nums.append(ClassNames.CLAPPING)

    # At the end of this code execution, pt will contain lists of lists, representing the 2D images.
    # Class nums contains a numerical representation of the classes, which can be used for training purposes.
    print('\n')
    print(len(pT), len(class_nums))
    return pT, class_nums

# Format the data from csv.
X, y = format_sequences(df, activity_count)
print("Formatting finished")
print(X)
print(y)

scaler = MinMaxScaler()
X = scaler.fit_transform(X) # Normalise the input data.
print("Fitting finished")

# Using a generative model, the input size should be reduced to the number of frames that will be used for classification purposes.

# Lets say that the input is of a fixed length between 4 and 12s for example. And that there are 3 possible input lengths, of 4s, 8s and 12s.
# We can refactor the input to a fixed number of "critical" frames, e.g. 8 frames to be passed on.

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




# Match the generated points to an activity class using the classifier.

# After the terminal point has been observed, start a new observation window, this observation window cannot be fed into the generator unless the number of frames is equal to or greater than the minimum.