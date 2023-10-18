import pandas as pd
import numpy as np
import csv

# Image dimensions are based on the radar configuration, these need to be set and changed inside of this file accordingly.
X_DIM = 143 # Corresponds to +/- 50 degrees.
Y_DIM = 35 # Corresponds to 5m clipped maximum range.
XY_DIM = X_DIM * Y_DIM

VERT_COMP_FACTOR = 5 # Vertical compression factor (y values) -> 7 pixels.
HORIZ_COMP_FACTOR = 13 # Horizontal compression factor (x values) -> 11 pixels.

ACTIVITY_COUNT_PER_PERSON = 60 # 20 samples of activity data was gathered for each activity, per person.
NUM_PARTICIPANTS = 1 # Participant data was gathered from 3 people in the research team.
 
def format_sequences(df, count, p_index):

    px_t = [] # Temporary list for x axis pixels.
    pp = np.zeros(round(X_DIM / HORIZ_COMP_FACTOR)) # This is a compression array, used for storing compressed pixel values.
    px = np.empty(round(X_DIM / HORIZ_COMP_FACTOR)) # This list will contain a row of pixels for a single frame.

    py = [] # This list will contain a set of pixel rows for a single frame.
    pt = [] # This list will contain the sequence of frames belonging to a single activity.
    pT = [] # Contains all frames for all sequences.

    xptr = 0 # Pointer for the end of the most recent row.
    yptr = 0 # Pointer for the end of the most recent frame.

    df = df.T # Transpose the data frame.

    # The loop below compresses each frame into a reduced format, then appends all of the compressed frames to a csv.
    for i in range(count):
        this_activity = df.iloc[i] # Get the next activity in the list of activities.
        this_class = (this_activity.iloc[0]) # Append the class.
        this_activity = this_activity.iloc[1:] # Remove the class before formatting the rest of the data.
        print(i) # Print the current activity index.

        # This will continue for the required number of frame iterations.
        for j in range(round(len(this_activity) / XY_DIM)):
            for k in range(Y_DIM):
                px_t = this_activity.iloc[xptr * X_DIM + yptr * XY_DIM : (xptr + 1) * X_DIM + yptr * XY_DIM].to_numpy()
                # If there are zeros where there shouldn't be, the lines below handles this so it doesn't break everything.
                if (len(px_t) != X_DIM) :
                    px_t = np.zeros(X_DIM)
                for m in range(round(X_DIM / HORIZ_COMP_FACTOR)):
                    px[m] = np.average(px_t[m * HORIZ_COMP_FACTOR : (m + 1) * HORIZ_COMP_FACTOR])
                pp = np.add(pp, px)
                if (k % VERT_COMP_FACTOR == 0):
                    pp = np.divide(pp, VERT_COMP_FACTOR)
                    py.append(np.array(pp)) # Append as array.
                    pp = np.zeros(round(X_DIM / HORIZ_COMP_FACTOR))
                xptr += 1
            pt.append(np.array(py)) # Append as array.
            py = [] # Reset py.
            yptr += 1
            xptr = 0
        pp = np.zeros(round(X_DIM / HORIZ_COMP_FACTOR)) # Reset after each inner loop.
        pt = np.array(pt) # Convert to array so we can operate on it.
        pt = pt.flatten('C') # Flatten in row-major order.
        pt = list(pt)
        pt.insert(0, this_class) # Put the class name back.
        pT.append(pt)
        yptr = 0
        pt = [] # Reset pt.
    
    # If we are only beginning to read in activities, write permissions should be granted in case the file doesn't exist yet, otherwise append.
    if p_index == 0:
        with open('C:\\GitHub\\mmWave-HAR\\Main\\Shared Resources\\reduced_data.csv', 'w', newline='') as file:
        # Using csv.writer to write the list to the CSV file.
            writer = csv.writer(file)
            writer.writerows(pT)
    else:
        with open('C:\\GitHub\\mmWave-HAR\\Main\\Shared Resources\\reduced_data.csv', 'a', newline='') as file:
        # Using csv.writer to write the list to the CSV file.
            writer = csv.writer(file)
            writer.writerows(pT)

# Paths to csv files containing activity data.
path_list = []
path_list.append('E:\\Data\\Output\\blank.csv')
path_list.append('E:\\Data\\Output\\clapping.csv')
path_list.append('E:\\Data\\Output\\jumping.csv')
path_list.append('E:\\Data\\Output\\standing.csv')
path_list.append('E:\\Data\\Output\\walking.csv')
path_list.append('E:\\Data\\Output\\waving.csv')

for path in path_list:

    # Read in the activity data for this file.
    activity_data = open(path).readlines()
    del activity_data[0] # Remove the headers at the top of the file.

    # Blank contains more sequences than the other files.
    if path == 'C:\\GitHub\\mmWave-HAR\\Main\\Shared Resources\\Blank.csv':
        activity_count = ACTIVITY_COUNT_PER_PERSON * NUM_PARTICIPANTS
    else:
        activity_count = ACTIVITY_COUNT_PER_PERSON

    df = pd.DataFrame()

    for i in range(activity_count):
        temp = activity_data[i].split(',')
        temp[-1] = temp[-1].split('\n')[0] # Remove the newline character at the end.
        activity_name = temp[0] 
        temp = temp[1:] # Remove the activity name (it is a string, not complex dtype, so separate it).
        cmplx = []
        for s in temp:
            s = s.replace('i', 'j')
            v = abs(complex(s)) # Use the modulus of the complex number.
            cmplx.append(v)
        cmplx.insert(0, activity_name)
        df[i] = cmplx

    # Print a small section of the dataframe to ensure things have been read in correctly.
    print(df[0])
    print("Activity data has been read in")

    # Data is in the format:
    # Activity sample 1: Class, Image 1, Image 2, Image 3, etc. 
    # Activity sample 2: Class, Image 1, Image 2, Image 3, etc.
    # etc.

    print(f"Activity count is {activity_count}\n")
    print(f"Path being used is: {path}")

    p_index = path_list.index(path) # Get index of the current path.
    format_sequences(df, activity_count, p_index) # Call the formatting function.