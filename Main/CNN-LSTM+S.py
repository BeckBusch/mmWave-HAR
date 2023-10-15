# CNN-LSTM Model for testing on gathered human activity data.
# Using pytorch implementation.
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from enum import Enum
from sklearn.utils import shuffle

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

# Seed setting.
RANDOM_SEED = 33
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

path = 'C:\\Users\\Samuel Mason\\Documents\\GitHub\\mmWave-HAR\\Main\\augmented_data.csv' # Path to the csv file with all of the activity data.
print(f"Path being used is: {path}")

# Read in the activity data, create a dataframe for it with the rows and columns transposed for optimisation.
activity_data = open(path).readlines()

# Total number of activities recorded.
activity_count = len(activity_data)

all_data = []

for i in range(activity_count):
    temp = activity_data[i].split(',')
    temp[-1] = temp[-1].split('\n')[0] # Remove the newline character at the end.
    activity_name = temp[0] 
    temp = temp[1:] # Remove the activity name (it is a string, not complex dtype, so separate it).
    temp.insert(0, activity_name)
    all_data.append(temp)

print("Activity data has been read in")

# Data is in the format:
# Activity sample 1: Class, Image 1, Image 2, Image 3, etc. 
# Activity sample 2: Class, Image 1, Image 2, Image 3, etc.
# etc.

print(f"Activity count is {activity_count}\n")

# Image dimensions are based on the compressed data size.
X_DIM = 11 
Y_DIM = 7 
XY_DIM = X_DIM * Y_DIM

def format_sequences(df, count):
    px = np.empty(X_DIM) # This list will contain a row of pixels for a single frame.
    py = [] # This list will contain a set of pixel rows for a single frame.
    pt = [] # This list will contain the sequence of frames belonging to a single activity.
    pT = [] # Contains all frames for all sequences.

    classes = [] # This list will contain the classes for each of the activity sequences.
    class_nums = [] # This list contains the enumerated classes.

    xptr = 0 # Pointer for the end of the most recent row.
    yptr = 0 # Pointer for the end of the most recent frame.

    for i in range(count):
        this_activity = df[i] # Get the next activity.
        classes.append(this_activity[0]) # Append the class.
        this_activity = this_activity[1:] # Remove the class before formatting the rest of the data.
        # Convert all values to floats.
        temp = []
        for j in range(len(this_activity)):
            temp.append(float(this_activity[j]))
        this_activity = temp
        # This will continue for the required number of frame iterations.
        for j in range(round(len(this_activity) / XY_DIM)):
            for k in range(Y_DIM):
                px = np.array(this_activity[xptr * X_DIM + yptr * XY_DIM : (xptr + 1) * X_DIM + yptr * XY_DIM])
                # If there are zeros where there shouldn't be, the below code handles this so it doesn't break everything.
                py.append(np.array(px)) # Append as array.
                xptr += 1
            pt.append(np.array(py)) # Append as array.
            py = [] # Reset py.
            yptr += 1
            xptr = 0
        pT.append(np.array(pt)) # Append as array.
        yptr = 0
        pt = [] # Reset pt.

    # Now we need to convert the classes to a numerical (enumerated) representation.
    for i in range(len(classes)):
        this_class = classes[i].split('_')[0] # Get the class name.
        if this_class == "empty":
            class_nums.append(ClassNames.EMPTY.value)
        elif this_class == "clapping":
            class_nums.append(ClassNames.CLAPPING.value)
        elif this_class == "jumpingjacks":
            class_nums.append(ClassNames.JUMPINGJACKS.value)
        elif this_class == "standing":
            class_nums.append(ClassNames.STANDING.value)
        elif this_class == "walking":
            class_nums.append(ClassNames.WALKING.value)
        elif this_class == "waving":
            class_nums.append(ClassNames.WAVING.value)

    # At the end of this code execution, pT will contain lists of lists, representing the 2D images.
    # Class nums contains a numerical representation of the classes, which can be used for training purposes.
    return np.array(pT), np.array(class_nums)

# Format the data from csv.
X, y = format_sequences(all_data, activity_count)
print("Formatting finished")
print(np.shape(X))

# Create an array to hold all the selected frames for each activity. Activity lengths can be arbitrary here, they should be uniformly scaled down to the correct size.
all_select_frames = []
for j in range(activity_count):
    # Create an array containing the average pixel value for each frame.
    averages = []
    this_activity = X[j]
    for k in range(len(this_activity)):
        this_frame = this_activity[k]
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
for j in range(activity_count):
    this_indices = all_select_frames[j]
    this_frames = []
    for k in range(TARGET_FRAMES):
        this_frames.append(X[j][this_indices[k]])
    all_frames.append(this_frames)

X = all_frames # Re-assign the variable name.

scaler = MinMaxScaler()
X = np.reshape(X, (activity_count, TARGET_FRAMES * Y_DIM * X_DIM))
X = scaler.fit_transform(X) # Normalise the input data.
X = np.reshape(X, (activity_count, TARGET_FRAMES, Y_DIM, X_DIM))
print("Fitting finished")

X, y = shuffle(X, y) # Shuffle the data!

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

def train_model(model, train_data, train_labels, val_data = None, val_labels = None, num_epochs = 100, verbose = 10, patience = 10):
    loss_fn = torch.nn.L1Loss() # L1 loss by default.
    optimiser = torch.optim.Adam(model.parameters(), lr = 0.001) # Default learning rate is 0.001.

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

        if val_data is not None:

            with torch.no_grad():

                val_loss = 0

                for val_idx, val_seq in enumerate(val_data):

                    model.reset_hidden_state() # Reset the hidden state with every sequence.

                    val_seq = torch.unsqueeze(val_seq, 0)
                    y_val_pred = model(val_seq)
                    val_step_loss = loss_fn(y_val_pred[0].float(), val_labels[val_idx])

                    val_loss += val_step_loss

            # Print the loss based on verbose value (pseudo-verbose).
            if t % verbose == 0:
                print(f'Epoch {t} train loss: {epoch_loss / len(train_data)} val loss: {val_loss / len(val_data)}')

                # Manual accuracy calculation.
                acc = 0
                for test_idx, test_seq in enumerate(X_test):

                    model.reset_hidden_state() # Reset the hidden state with every sequence.

                    test_seq = torch.unsqueeze(test_seq, 0)
                    y_test_pred = model(test_seq)
                    
                    if (torch.round(y_test_pred).detach().numpy()[0] == y_test[test_idx].numpy()):
                        acc += 1

                acc /= test_idx
                print(f'Testing accuracy is {acc}\n')

            # Can add early stopping if wanted, not using currently.

    return model

model = CNNLSTM(
    n_features = 15,
    n_hidden = 8,
    seq_len = TARGET_FRAMES,
    n_layers = 1
)

model = train_model(
    model, 
    X_train,
    y_train,
    X_val,
    y_val,
    num_epochs = 150,
    verbose = 10,
    patience = 50
)

# Save the model once training is complete.
torch.save(model, 'C:\\Users\\Samuel Mason\\Documents\\GitHub\\mmWave-HAR\\Main\\CNN-LSTM+S_model.pth')
print("Model saved successfully")
