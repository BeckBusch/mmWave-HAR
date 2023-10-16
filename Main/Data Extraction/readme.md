# Main\Data Extraction\
## \Final Capture
Files exported from mmWave Studio containing information about the radar setup and recording metadata. These files are used in the processing of the raw binaries
### FullProcessRangeAngle.m
MATLAB script that convertes the raw radar binaries into a sanitised, compressed, and cropped CSV file of data.
### rawDataReader.m
Modified version of a script provided by Texas Instruments that converts raw radar binaries into matlab matrices. The scripts job is to fill missing frames of the recording with zero values, and to properly order any out-of-order frames