import numpy as np
import pandas as pd

# We want to augment the data by producing synthetic samples, in order to increase the data that we can train on.

# For each activity of 6 seconds in length, we want to produce samples of 2, 3, 4, 5 and 6s, with 5, 4, 3, 2, 1 samples respectively.
# This totals 15 samples per sample, so we can get a x15 increase in the size of the dataset.

# Defines
FRAMES_PER_SECOND = 25 # From mmWave Studio.
SAMPLE_LENGTH = 6 # Sample length in seconds.

path = 'C:\\Data\\reduced_data.csv' # Path to the csv file with all of the activity data.
# The path being used here is for the reduced data i.e. the data that has had vertical (and potentially horizontal) image compression done.
print(f"Path being used is: {path}")

# Read in the activity data.
activity_data = open(path).readlines()
del activity_data[0]
print("Activity data has been read in")

# Total number of activities recorded.
activity_count = len(activity_data)

for p in range(activity_count):
    this_sample = activity_data[p] # Get a new activity sample to augment.
    for i in range(5):
        n = i + 1
        for j in range(n):
            # The above loops produce an inner loop of 1, 2, 3, 4, 5 loops.
            x = np.random.uniform(0.0, 1.0) # Generate a uniformly distributed random number between 0 and 1 to be used to offset the subsample.
            start_seconds = n + x # Point, in seconds, where we start clipping from.
            duration = SAMPLE_LENGTH - i # Point, in seconds, where we end the subsample clipping.
            start_ptr = round(FRAMES_PER_SECOND * start_seconds) # Pointer to the starting frame. 
            end_ptr = round(FRAMES_PER_SECOND * duration + start_ptr) # Pointer to the ending frame.
            new_sample = this_sample[start_ptr : end_ptr]
            # Append the array containing the newly generated sample to the csv file.
            pd.DataFrame(new_sample).to_csv("augmented_data.csv")

# Once we have done this for every data sample and written to the output csv, we are done.
