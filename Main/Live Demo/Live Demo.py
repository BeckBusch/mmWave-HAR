import shutil
import os
import subprocess   

recordingBinSource = "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\adc_data_Raw_0.bin"
destinationName = "C:\\GitHub\\mmWave-HAR\\Main\\Live Demo\\LiveDemoRecording.bin"

# Relocate the bin file
while(True):
    try:
        shutil.move(recordingBinSource, destinationName)
        break

    except FileNotFoundError:
        input("File not found :(")

# Run matlab data extraction
os.system("matlab -nosplash -nodesktop -r \"run(\'C:\\GitHub\\mmWave-HAR\Main\\Data Extraction\\FullProcessRangeAngle.m\'); exit;\"")

# Wait for data extraction to finish
while not os.path.exists("C:\GitHub\mmWave-HAR\Main\Live Demo\done"):
    pass

# Compress the data
os.system("python \"C:\\GitHub\\mmWave-HAR\\Main\\Preprocessing Scripts\\data_compressor.py\"")

# Run the model
os.system("python \"C:\\GitHub\\mmWave-HAR\\Main\\Classification Scripts\\model_eval.py\"")