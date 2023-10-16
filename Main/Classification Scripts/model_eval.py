# Load the trained model of choice

# Import statements
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from torch import nn
from enum import Enum
import joblib

# Class names for each of the 5 (+1) activity classes.
class ClassNames(Enum):
    STANDING = 0
    WALKING = 1
    CLAPPING = 2
    WAVING = 3
    JUMPINGJACKS = 4
    EMPTY = 5

TARGET_FRAMES = 16 # Input size to the classifier that we want to reduce to.
INTERPOLATION_FRAMES = TARGET_FRAMES + 2 # Need 2 additional frames, 1 for final frame which is removed, and first frame is not used.
SAMPLE_LENGTH = 6 # Sample length in seconds.
MIN_PASS_LENGTH = 2 # Minimum length of an activity sequence to make inferences from (in seconds).
FRAMES_PER_SECOND = 25 # Number of frames in one second of radar data.
STEP_SIZE = 5 # Number of frames to step through by - corresponds to ~0.5s.

# Path to the model and activity to be evaluated.
PATH = '..\\Trained Models\\CNN-LSTM+S_model.pth'
activity_path = '..\\Shared Resources\\eval.csv'

class CNNLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(CNNLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers
        # 2D CNN layer.
        self.c = nn.Conv2d(in_channels = TARGET_FRAMES, out_channels = TARGET_FRAMES, kernel_size = 3, stride = 2)
        self.lstm = nn.LSTM(
            input_size = n_features,
            hidden_size = n_hidden,
            num_layers = n_layers
        )
        self.linear = nn.Linear(in_features = n_hidden, out_features = 1)

    # IMPORTANT! This assumes that the sequences input from the csv are of a uniform length.
    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(1, 1, self.n_hidden),
            torch.zeros(1, 1, self.n_hidden)
        )

    def forward(self, seq):
        seq = self.c(seq)
        lstm_out, self.hidden = self.lstm(
            seq.view(TARGET_FRAMES, -1),
        )
        last_time_step = lstm_out.view(self.seq_len, len(seq), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred

# Read in the activity data.
seq = open(activity_path).readline()
print("Activity data has been read in")

# Format the activity data frames.
seq = seq.split(',')
seq[-1] = seq[-1].split('\n')[0] # Remove the newline character at the end.
activity_name = seq[0] 
seq = seq[1:] # Remove the activity name (it is a string, not complex dtype, so separate it).
seq.insert(0, activity_name)

# Image dimensions are based on the radar configuration, these need to be set and changed inside of this file accordingly.
X_DIM = 11 #36 # For example.
Y_DIM = 7 #18 # For example.
XY_DIM = X_DIM * Y_DIM

def format_seq(seq):
    px = np.empty(X_DIM) # This list will contain a row of pixels for a single frame.
    py = [] # This list will contain a set of pixel rows for a single frame.
    pt = [] # This list will contain the sequence of frames belonging to a single activity.

    xptr = 0 # Pointer for the end of the most recent row.
    yptr = 0 # Pointer for the end of the most recent frame.

    class_name = seq[0] # Append the class.
    seq = seq[1:] # Remove the class before formatting the rest of the data.
    # Convert all values to floats.
    temp = []
    for j in range(len(seq)):
        temp.append(float(seq[j]))
    seq = temp
    # This will continue for the required number of frame iterations.
    for j in range(round(len(seq) / XY_DIM)):
        for k in range(Y_DIM):
            px = np.array(seq[xptr * X_DIM + yptr * XY_DIM : (xptr + 1) * X_DIM + yptr * XY_DIM])
            # If there are zeros where there shouldn't be, the below code handles this so it doesn't break everything.
            py.append(np.array(px)) # Append as array.
            xptr += 1
        pt.append(np.array(py)) # Append as array.
        py = [] # Reset py.
        yptr += 1
        xptr = 0
    yptr = 0

    # Need to convert the class to a numerical (enumerated) representation.
    class_names = [class_name.split('_')[0], class_name.split('_')[1]]

    class_nums = []
    for this_class in class_names:
        if this_class == "empty":
            class_num = ClassNames.EMPTY.value
        elif this_class == "clapping":
            class_num = ClassNames.CLAPPING.value
        elif this_class == "jumpingjacks":
            class_num = ClassNames.JUMPINGJACKS.value
        elif this_class == "standing":
            class_num = ClassNames.STANDING.value
        elif this_class == "walking":
            class_num = ClassNames.WALKING.value
        elif this_class == "waving":
            class_num = ClassNames.WAVING.value
        class_nums.append(class_num)

    # At the end of this code execution, pt will contain the frames for this activity.
    # Class nums is a numerical representation of the class(es).
    return np.array(pt), np.array(class_nums)

# Format the data from csv.
X, y = format_seq(seq)
print("Formatting finished")
print(np.shape(X))

all_select_frames = []
# Cut up the activity data to produce a sample for each pass.
pass_count = round(np.floor(FRAMES_PER_SECOND * (SAMPLE_LENGTH - MIN_PASS_LENGTH) / STEP_SIZE))
for j in range(pass_count):

    # Only fetch the frames for this pass.
    this_pass_frames = X[0:FRAMES_PER_SECOND * MIN_PASS_LENGTH + j * STEP_SIZE]

    # Create an array containing the average pixel value for each frame.
    averages = []
    for k in range(len(this_pass_frames)):
        this_frame = this_pass_frames[k]
        this_avg = np.average(this_frame) # Get the average pixel value for this frame.
        averages.append(this_avg)
    max_frames = []
    select_frames = []
    # We want half of the target frames to be peak values, hence divide by 2.
    for n in range(round(INTERPOLATION_FRAMES / 2)):
        section_size = round(np.floor(len(averages) / (INTERPOLATION_FRAMES / 2)))
        current_max_index = averages.index(max(averages[n * section_size : (n + 1) * section_size]))
        max_frames.append(current_max_index)
    # After the below loop, we should end up with TARGET_FRAMES number of frames in select_frames
    for n in range(round(INTERPOLATION_FRAMES / 2) - 1):
        current_max_index = max_frames[n]
        current_min_index = round((max_frames[n] + max_frames[n + 1]) / 2)
        # Attach the next pair of min/max frames.
        select_frames.append(current_max_index)
        select_frames.append(current_min_index)
    all_select_frames.append(select_frames)

# Append the selected frames based on the stored indices.
all_frames = []
for j in range(pass_count):
    this_indices = all_select_frames[j]
    this_frames = []
    for k in range(TARGET_FRAMES):
        this_frames.append(X[this_indices[k]])
    all_frames.append(this_frames)

# Transform to a tensor.
def make_Tensor(array):
    return torch.from_numpy(array).float()

# MinMaxScaler.
scaler_fname = "..\\Trained Models\\scaler.pkl"
scaler = joblib.load(scaler_fname)

# Load in the saved model.
model = torch.load(PATH)
print("Model loaded successfully!")
model.eval()

preds = []
pred_names = []
for _ in range(pass_count):

    X = all_frames[_] # Frames for the current pass.

    X = np.reshape(X, (1, TARGET_FRAMES * Y_DIM * X_DIM))
    X = scaler.transform(X) # Normalise the input data.
    X = np.reshape(X, (1, TARGET_FRAMES, Y_DIM, X_DIM))
    X = make_Tensor(X)

    # Make a prediction for this pass.
    with torch.no_grad():
        model.reset_hidden_state()
        y_test_pred = model(X)
        y_pred = torch.flatten(y_test_pred).item()
        preds.append(y_pred)
        pred_names.append(ClassNames(round(y_pred)).name)

y = make_Tensor(y)
y1 = torch.flatten(y[0]).item() # First class.
# y2 = torch.flatten(y[1]).item()
print(f"Actual activity class is: {ClassNames(y1).name}")

print(preds)
print(pred_names)

