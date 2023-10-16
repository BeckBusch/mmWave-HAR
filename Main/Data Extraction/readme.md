# Main\Data Extraction\
## \Final Capture
Files exported from mmWave Studio containing information about the radar setup and recording metadata. These files are used in the processing of the raw binaries.

This script first indexes all binary files in a given directory, and begins to process them. Each binary final is pre-processed and converted into matrices by rawDataReader.m. These matrices are then used to construct a Radar Data Cube object for each frame of the recording. the radar data cube is beamformed into a range-angle response graph, which is cropped to a specified range and angle limits. After each frame of every recording has been processed, the results are saved to a CSV file.
### FullProcessRangeAngle.m
MATLAB script that convertes the raw radar binaries into a sanitised, compressed, and cropped CSV file of data.
### rawDataReader.m
Modified version of a script provided by Texas Instruments that converts raw radar binaries into matlab matrices. The scripts job is to fill missing frames of the recording with zero values, and to properly order any out-of-order frames