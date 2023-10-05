# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import mmwave
from mmwave.dataloader import DCA1000
import numpy as np

# Radar specific parameters
NUM_RX = 4

# Data specific parameters
NUM_CHIRPS = 128
NUM_ADC_SAMPLES = 256
RANGE_RESOLUTION = .0488
DOPPLER_RESOLUTION = 0.0806
NUM_FRAMES = 5

# DSP processing parameters
SKIP_SIZE = 4
ANGLE_RES = 1
ANGLE_RANGE = 90
ANGLE_BINS = (ANGLE_RANGE * 2) // ANGLE_RES + 1
BINS_PROCESSED = 112

# Read in adc data file
load_data = True
if load_data:
    adc_data = np.fromfile('./Testing/5FrameRec/5Frame.bin', dtype=np.uint16)    
    adc_data = adc_data.reshape(NUM_FRAMES, -1)
    all_data = np.apply_along_axis(DCA1000.organize, 1, adc_data, num_chirps=NUM_CHIRPS, num_rx=NUM_RX, num_samples=NUM_ADC_SAMPLES)

range_bins = mmwave.dsp.range_processing(all_data, window_type_1d=None, axis=-1)

range_bins = np.trunc(range_bins)

for i in range(0, NUM_FRAMES-1):
    print(range_bins[i][1][0][1], range_bins[i][1][2][1], range_bins[i][1][3][1], range_bins[i][1][1][1])
