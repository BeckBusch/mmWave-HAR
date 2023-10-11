import pandas as pd
import numpy as np
from enum import Enum
import csv

class ClassNames(Enum):
    JUMPING = 0
    WALKING = 1
    CLAPPING = 2

# Image dimensions are based on the radar configuration, these need to be set and changed inside of this file accordingly.
X_DIM = 143 # For example.
Y_DIM = 35 # For example.
XY_DIM = X_DIM * Y_DIM

VERT_COMP_FACTOR = 5 # Vertical compression factor (y values) -> 7 pixels.
HORIZ_COMP_FACTOR = 13 # Horizontal compression factor (x values) -> 11 pixels.

def format_sequences(df, count, p_index):
    px_t = [] # Temporary list for x axis pixels.
    px = np.empty(round(X_DIM / HORIZ_COMP_FACTOR)) # This list will contain a row of pixels for a single frame.
    py = [] # This list will contain a set of pixel rows for a single frame (one frame).
    pt = [] # This list will contain the sequence of frames.

    pT = [] # Contains all frames.

    pp = np.zeros(round(X_DIM / HORIZ_COMP_FACTOR)) # This is a compression array, used for storing compressed pixel values.

    xptr = 0 # Pointer for the end of the most recent row.
    yptr = 0 # Pointer for the end of the most recent frame.

    df = df.T # Transpose the data frame.
    #print(df[0])
    #print(df[1])

    # We need to start at 1 because the first values are not reliable.
    for i in range(count):
        this_activity = df.iloc[i] # Get the next activity.
        #print(this_activity)
        #print(this_activity.iloc[0])
        this_class = (this_activity.iloc[0]) # Append the class.
        this_activity = this_activity.iloc[1:] # Remove the class before formatting the rest of the data.
        print(i)
        # print(this_activity[:20])
        # print("\n\n")
        #print(this_activity)
        # This will continue for the required number of frame iterations.
        for j in range(round(len(this_activity) / XY_DIM)):
            for k in range(Y_DIM):
                px_t = this_activity.iloc[xptr * X_DIM + yptr * XY_DIM : (xptr + 1) * X_DIM + yptr * XY_DIM].to_numpy()
                # If there are zeros where there shouldn't be, the below code handles this so it doesn't break everything.
                # print(px[:10])
                # print(pp[:10])
                if (len(px_t) != X_DIM) :
                    px_t = np.zeros(X_DIM)
                for m in range(round(X_DIM / HORIZ_COMP_FACTOR)):
                    px[m] = np.average(px_t[m * HORIZ_COMP_FACTOR : (m + 1) * HORIZ_COMP_FACTOR])
                pp = np.add(pp, px)
                # print(pp[:10])
                # print("\n\n")
                if (k % VERT_COMP_FACTOR == 0):
                    pp = np.divide(pp, VERT_COMP_FACTOR)
                    py.append(np.array(pp)) # Append as array.
                    pp = np.zeros(round(X_DIM / HORIZ_COMP_FACTOR))
                xptr += 1
            # print(py[:10])
            # print("\n\n")
            pt.append(np.array(py)) # Append as array.
            py = [] # Reset py.
            yptr += 1
            xptr = 0
        pp = np.zeros(round(X_DIM / HORIZ_COMP_FACTOR)) # Reset after each inner loop.
        pt = np.array(pt) # Convert to array so we can operate on it.
        pt = pt.flatten('C') # Flatten in row-major order.
        pt = list(pt)
        pt.insert(0, this_class) # Put the class name back.
        # np.insert(pt, 0, this_class)

        pT.append(pt)
        yptr = 0
        # print(pt[:10])
        # print("\n\n")
        pt = [] # Reset pt.

        # pd.DataFrame(pt).to_csv("reduced_data.csv") # Append to reduced data csv.
    
    if p_index == 0:
        with open('reduced_data.csv', 'w', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
            writer = csv.writer(file)
            writer.writerows(pT) # Use writerow for single list
    else:
        with open('reduced_data.csv', 'a', newline='') as file:
        # Step 4: Using csv.writer to write the list to the CSV file
            writer = csv.writer(file)
            writer.writerows(pT) # Use writerow for single list

path_list = [] # Path to the csv file with all of the activity data (change this to get different activity data if individual files are too big).
path_list.append('C:\\Data\\output.csv')

for path in path_list:

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

    print(f"Path being used is: {path}")

    # Get index of the current path
    p_index = path_list.index(path)

    format_sequences(df, activity_count, p_index) # Actually call the function.