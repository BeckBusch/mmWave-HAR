import csv

# This file is for renaming activity data to be consistent with the naming scheme.

FRAMES_PER_SECOND = 25 # From mmWave Studio.
SAMPLE_LENGTH = 6 # Sample length in seconds.
IMAGE_PIXELS = 77 # Pixels in a single reduced image.

path = '..\\Shared Resources\\to_rename_data.csv' # Path to the csv file with all of the activity data.
print(f"Path being used is: {path}")

# Read in the activity data, create a dataframe for it with the rows and columns transposed for optimisation.
activity_data = open(path).readlines()
print("Activity data has been read in")

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

all_samples = []

for p in range(activity_count):
    this_sample = all_data[p] # Get an activity sample to rename.
    class_name = this_sample[0]
    this_sample = this_sample[1:]

    this_class = class_name.split('_')[0] # Get the class name (old).
    this_class_index = class_name.split('_')[1] # Get activity index belonging to participant for this class.

    if this_class == "blank":
        new_class_name = "empty" + "_00" + "_6s_" + this_class_index # Empty doesn't belong to any participant number
    elif this_class == "kevinclapping":
        new_class_name = "clapping" + "_01" + "_6s_" + this_class_index
    elif this_class == "KevinJacks":
        new_class_name = "jumpingjacks" + "_01" + "_6s_" + this_class_index
    elif this_class == "kevinStanding":
        new_class_name = "standing" + "_01" + "_6s_" + this_class_index
    elif this_class == "kevinWalking":
        new_class_name = "walking" + "_01" + "_6s_" + this_class_index
    elif this_class == "kevinWaving":
        new_class_name = "waving" + "_01" + "_6s_" + this_class_index
    elif this_class == "samClapping":
        new_class_name = "clapping" + "_02" + "_6s_" + this_class_index
    elif this_class == "SamJacks":
        new_class_name = "jumpingjacks" + "_02" + "_6s_" + this_class_index
    elif this_class == "samStanding":
        new_class_name = "standing" + "_02" + "_6s_" + this_class_index
    elif this_class == "SamWalking":
        new_class_name = "walking" + "_02" + "_6s_" + this_class_index
    elif this_class == "samWaving":
        new_class_name = "waving" + "_02" + "_6s_" + this_class_index
    elif this_class == "samWalking":
        new_class_name = "walking" + "_02" + "_6s_" + this_class_index
    elif this_class == "beckClapping":
        new_class_name = "clapping" + "_03" + "_6s_" + this_class_index
    elif this_class == "BeckJacks":
        new_class_name = "jumpingjacks" + "_03" + "_6s_" + this_class_index
    elif this_class == "beckStanding":
        new_class_name = "standing" + "_03" + "_6s_" + this_class_index
    elif this_class == "beckWalking":
        new_class_name = "walking" + "_03" + "_6s_" + this_class_index
    elif this_class == "beckWaving":
        new_class_name = "waving" + "_03" + "_6s_" + this_class_index

    this_sample.insert(0, new_class_name) # Attach the new class name.
    all_samples.append(this_sample)

# Once we have done this for every data sample and written to the output csv, we are done.
with open('..\\Shared Resources\\reduced_data.csv', 'w', newline='') as file:
# Using csv.writer to write the list to the CSV file.
    writer = csv.writer(file)
    writer.writerows(all_samples)
