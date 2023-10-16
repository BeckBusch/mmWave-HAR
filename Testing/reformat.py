import numpy as np

# a = np.fromfile("s_walk_1.bin")

numADCSamples = 256; # number of ADC samples per chirp
numADCBits = 16; # number of ADC bits per sample
numRX = 4; # number of receivers
numLanes = 2; # do not change. number of lanes is always 2
isReal = 0; # set to 1 if real only data, 0 if complex data0

# read .bin file
adc_data = np.fromfile("s_walk_1.bin", dtype = np.int16)
adc_data = adc_data.astype(np.float16)

adc_data /= 100 # Scale based on number of bits used...

print(adc_data[0:10])

fsize = len(adc_data)
print(fsize)

if not isReal == 1:
    print(0)
    numChirps = int(fsize / 2 / numADCSamples / numRX)
    print(numChirps)
    LVDS = np.zeros((1, int(fsize / 2)), dtype = np.complex_)
    # Combine real and imaginary part into complex data
    # Read in file: 2I is followed by 2Q
    counter = 0
    for i in range(int(fsize / 4)):
        # Need to multiply i by 4 so we index through correctly.
        LVDS[0, counter] = adc_data[i * 4] + 1j * adc_data[i * 4 + 2]
        LVDS[0, counter + 1] = adc_data[i * 4 + 1] + 1j * adc_data[i * 4 + 3]
        counter += 2
        if counter % 10000 == 1:
            print(counter)

    LVDS = np.reshape(LVDS, (numADCSamples * numRX, numChirps))
    # Each row is data from one chirp.
    LVDS = np.transpose(LVDS)

print(LVDS.shape)

# Organise data per RX.
out_data = np.zeros((numRX, numChirps * numADCSamples), dtype = np.complex_)
for row in range(numRX):
    for i in range(numChirps):
        out_data[row, i * numADCSamples + 1 : (i + 1) * numADCSamples] = LVDS[i, row * numADCSamples + 1 : (row + 1) * numADCSamples]

# This should be correct hopefully...
print(out_data.shape)
print(out_data[0:4, 1])

# adcData = fread(fid, 'int16');
# # if 12 or 14 bits ADC per sample compensate for sign extension
# if numADCBits ~= 16
#     l_max = 2^(numADCBits-1)-1;
#     adcData(adcData > l_max) = adcData(adcData > l_max) - 2^numADCBits;
# end
# fclose(fid);
# fileSize = size(adcData, 1);
# # real data reshape, filesize = numADCSamples*numChirps
# if isReal
#     numChirps = fileSize/numADCSamples/numRX;
#     LVDS = zeros(1, fileSize);
#     #create column for each chirp
#     LVDS = reshape(adcData, numADCSamples*numRX, numChirps);
#     #each row is data from one chirp
#     LVDS = LVDS.';
# else
#     # for complex data
#     # filesize = 2 * numADCSamples*numChirps
#     numChirps = fileSize/2/numADCSamples/numRX;
#     LVDS = zeros(1, fileSize/2);
#     #combine real and imaginary part into complex data
#     #read in file: 2I is followed by 2Q
#     counter = 1;
#     for i=1:4:fileSize-1
#     LVDS(1,counter) = adcData(i) + sqrt(-1)*adcData(i+2); LVDS(1,counter+1)...
#         = adcData(i+1)+sqrt(-1)*adcData(i+3); counter = counter + 2;
#     end
#     # create column for each chirp
#     LVDS = reshape(LVDS, numADCSamples*numRX, numChirps);
#     #each row is data from one chirp
#     LVDS = LVDS.';
# end
# #organize data per RX
# adcData = zeros(numRX,numChirps*numADCSamples);
# for row = 1:numRX
#     for i = 1: numChirps
#         adcData(row, (i-1)*numADCSamples+1:i*numADCSamples) = LVDS(i, (row-...
#         1)*numADCSamples+1:row*numADCSamples);
#     end
# end
# # return receiver data
# retVal = adcData;