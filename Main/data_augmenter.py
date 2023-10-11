import numpy as np
import pandas as pd
import csv

# We want to augment the data by producing synthetic samples, in order to increase the data that we can train on.

# For each activity of 6 seconds in length, we want to produce samples of 2, 3, 4, 5 and 6s, with 5, 4, 3, 2, 1 samples respectively.
# This totals 15 samples per sample, so we can get a x15 increase in the size of the dataset.

# Defines
FRAMES_PER_SECOND = 25 # From mmWave Studio.
SAMPLE_LENGTH = 6 # Sample length in seconds.
IMAGE_PIXELS = 77 # Pixels in a single image.

path = 'C:\\Users\\Samuel Mason\\Documents\\GitHub\\mmWave-HAR\\Main\\reduced_data.csv' # Path to the csv file with all of the activity data.
# The path being used here is for the reduced data i.e. the data that has had vertical (and potentially horizontal) image compression done.
print(f"Path being used is: {path}")

# Read in the activity data, create a dataframe for it with the rows and columns transposed for optimisation.
activity_data = open(path).readlines()
del activity_data[0]
print("Activity data has been read in")

# Total number of activities recorded.
activity_count = len(activity_data)
all_data = []

for i in range(activity_count):
    temp = activity_data[i].split(',')
    temp[-1] = temp[-1].split('\n')[0] # Remove the newline character at the end.
    activity_name = temp[0] 
    temp = temp[1:] # Remove the activity name (it is a string, not complex dtype, so separate it).
    # cmplx = []
    # for s in temp:
    #     cmplx.append(s)
    temp.insert(0, activity_name)
    # print(temp)
    all_data.append(temp)

all_samples = []

for p in range(activity_count):
    this_sample = all_data[p] # Get a new activity sample to augment.
    class_name = this_sample[0]
    this_sample = this_sample[1:]
    # print(this_sample)
    # print(len(this_sample))
    # print('\n\n')
    for i in range(SAMPLE_LENGTH):
        for j in range(i):
            # The above loops produce an inner loop of 1, 2, 3, 4, 5 loops.
            if j == 0:
                start_seconds = 0 # For the maximum length we can't clip.
            else:
                start_seconds = np.random.uniform(0.0, 1.0) + j - 1 # Generate a uniformly distributed random number between 0 and 1 to be used to offset the subsample.
            # print(start_seconds)
            duration = SAMPLE_LENGTH - i # Point, in seconds, where we end the subsample clipping.
            start_ptr = round(FRAMES_PER_SECOND * start_seconds) # Pointer to the starting frame. 
            #print(start_ptr)
            end_ptr = round(FRAMES_PER_SECOND * duration + start_ptr) # Pointer to the ending frame.
            #print(end_ptr)
            new_sample = this_sample[start_ptr * IMAGE_PIXELS : end_ptr * IMAGE_PIXELS]
            # print(f'START {start_ptr * IMAGE_PIXELS} END {end_ptr * IMAGE_PIXELS}\n')
            new_sample = list(new_sample)
            # print(this_sample[0])
            new_sample.insert(0, class_name)
            for q in range(IMAGE_PIXELS * FRAMES_PER_SECOND * (SAMPLE_LENGTH - duration)):
                new_sample.append(0) # Append zeros to fill the rest of the row - this way the rows will have uniform length.
            # print(new_sample[2])
            # Append the array containing the newly generated sample to the csv file.

            all_samples.append(new_sample)

with open('augmented_data.csv', 'w', newline='') as file:
# Step 4: Using csv.writer to write the list to the CSV file
    writer = csv.writer(file)
    writer.writerows(all_samples) # Use writerow for single list

# Once we have done this for every data sample and written to the output csv, we are done.
