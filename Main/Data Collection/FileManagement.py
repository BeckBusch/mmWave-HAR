import easygui
import os
import shutil


recordingBinSource = "C:\\ti\\mmwave_studio_02_01_01_00\\mmWaveStudio\\PostProc\\adc_data_Raw_0.bin"
recordingBinDestination = "E:\\Records\\Incoming\\"

welcomeMsg = """What is the name for this Session?

This name will be used for labeling the files and is case sensitive.
Please ensure that the given session name is unique enough to identify the recordings from this session

Example: Starjumps_Joe_6sec"""
recordingMsg = """Create a recording in mmWave studio, then press done.

The recording will be moved to the external drive and renamed to match the session.

If there are no further recordings in this session, press Session Finished"""
sessionContinueMsg = "Done"
sessionEndMsg = "Session Finished"


# receive session name
sessionName = easygui.enterbox(welcomeMsg, "Welcome")

inProgress = True # I guess this is just a worse while(true):break but idk it works
recordingCount = 0

while inProgress != False:
    # Prompt the user
    check = easygui.buttonbox(recordingMsg, "Recording In Progress...", [sessionContinueMsg, sessionEndMsg])

    # Check if the answer was finished
    if check == sessionEndMsg:
        inProgress = False
        continue
    
    # Otherwise we carry on
    recordingCount += 1
    destinationName = recordingBinDestination + sessionName + "_" + str(recordingCount) + ".bin"
    
    try:
        shutil.move(recordingBinSource, destinationName)
    except FileNotFoundError:
        recordingCount -= 1
        easygui.buttonbox("Recording not found :(", "File Missing", ["ok"])

