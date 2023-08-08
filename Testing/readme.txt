
Overview
==========
This Matlab script is to post process a *.setup.json file generated from mmwave studio with captured bin files.
It generates mat files for raw ADC data and radar cube data.


Before Test
==============
1. Install mmwave studio on Windows
2. Install Matlab 2017a or later on windows
3. Setup radar device and data card with a corner reflector at a reasonable distance.

Execution steps
================

1. Following mmwave Studio user guide in section "Capturing Raw ADC Data", configure/capture/export JSON files with captured bin files.

2. From Matlab prompt run post processing script, e.g.
	
   a. Generate raw ADC data and radarCube data only:
    
       rawDataReader('C:\jsonScript\awr14\awr14xx.setup.json','adcData', 'radarCube', 0)

   b. Debug plot only

       rawDataReader('C:\jsonScript\awr14\awr14xx.setup.json','', '', 1)

   Note: Exported Files from example a can be used for verification script.

3. Run verification code to test the output of generated radar cube data file.

   verify_data('radarCube.mat');

   This script will print out the range profile peak information for verification. 
   Please put a corner reflector before capturing the rad ADC data in step to get a reasonable result.


Notes
=======

1. *.setup.json file contains additional files such as *.mmwave.json file and bin files. If files are copied to a new location, please update the path in *.setup.json file properly. 

2. If the raw data file is big, it may take some file to finish generating the mat files. 

