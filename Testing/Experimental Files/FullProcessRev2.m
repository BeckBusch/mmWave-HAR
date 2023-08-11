clear;

load STAPExampleData;


% define sensor array
fc = 77e9;
c = physconst('LightSpeed');
lambda = c/fc;
Nt = 3;
Nr = 4;
dt = lambda;
dr = lambda/2;
txarray = phased.ULA(Nt,dt);
rxarray = phased.ULA(Nr,dr);

% setup file read vars
setupExportPath = 'RadarCapture/setupExport.setup.json';
adcSavePath = '';
radarCubeSavePath = 'RadarCapture/NewRadarCube';
binFilePath = 'RadarCapture/adc_data.bin';

% load radar cube from bin
cube = rawDataReader(setupExportPath, adcSavePath, radarCubeSavePath, 0, binFilePath);

% isolate single frame in toolbox format: adc Samples, Rx, chirps
frame = cube.data;
frame = cell2mat(frame(100));
frame = permute(frame, [3 2 1]); % frame(:, 1, :)
frame = shiftdim(frame(200,:,:)); % isolate single sample? i.e. rx by chirp for one sample

response = phased.AngleDopplerResponse( ...
    'SensorArray',rxarray, ... % antenna array should be easy
    'OperatingFrequency',cube.rfParams.startFreq, ...
    'PropagationSpeed',c, ...
    'PRF',cube.rfParams.sampleRate, ...
    'NumAngleSamples',cube.rfParams.numRangeBins,...
    'NumDopplerSamples',cube.rfParams.numDopplerBins);

[resp,ang_grid,dop_grid] = response(frame);
contour(ang_grid,dop_grid,abs(resp))
xlabel('Angle')
ylabel('Doppler')









