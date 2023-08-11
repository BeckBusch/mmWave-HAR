%%% Range Angle Process Script
%%% Beck Busch & Samuel Mason
%%% 

% Clear Workspace
clear;

% Setup .bin File List
sourceDir = 'RadarBins/';
list = dir(strcat(sourceDir, '*.bin'));
filenames = {list.name};
fileCount = length(filenames);

% Create Sensor Array
fc = 77e9;
c = physconst('LightSpeed');
lambda = c/fc;
Nr = 4;
dr = lambda/2;
rxarray = phased.ULA(Nr,dr);

% Create range-angle responce object
        rngangresp = phased.RangeAngleResponse(...
            'SensorArray',rxarray, ...
            'OperatingFrequency',fc,...
            'SampleRate',10e8, ...
            'PropagationSpeed',c);

% Setup File Paths
setupExportPath = 'FinalCapture/setupExport.setup.json';
adcSavePath = '';
radarCubeSavePath = ''; %'RadarCapture/NewRadarCube';

% hard coded sizes
hardFrameCount = 150;
hardClipSize = 5005;

finalExport = cell(fileCount, hardFrameCount+1);

% Iterate over all files
for CurrentFileNumber=1:fileCount

    % Select current filename
    binFilePath = filenames(CurrentFileNumber);
    binFilePath = binFilePath(1);

    finalExport(CurrentFileNumber, 1) = binFilePath;

    binFilePath = strcat(sourceDir, binFilePath);
    binFilePath = cell2mat(binFilePath(1));

    % load radar cube from bin
    cube = rawDataReader(setupExportPath, adcSavePath, radarCubeSavePath, 0, binFilePath);
    frameMat = cube.data;
    
    % Iterare over all frames in bin file
    for currentFrameNumber=1:cube.dim.numFrames
    
        % isolate single frame in toolbox format
        frame = cell2mat(frameMat(currentFrameNumber));
        frame = permute(frame, [3 2 1]);
        frame = shiftdim(frame(:, :, 1)); % isolate single whatever
        frame = double(frame);
        
        % Process range-angle responce object over the current frame
        [resp,rng_grid,ang_grid] = rngangresp(frame,[1;1]);
    
        lowerAngleLimit = find(ang_grid >= (-50));
        lowerAngleLimit = lowerAngleLimit(1);
        upperAngleLimit = find(ang_grid >= (50));
        upperAngleLimit = upperAngleLimit(1);
        rangeLimit = find(rng_grid >= 5);
        rangeLimit = rangeLimit(1);

        radarClip = resp(1:rangeLimit,lowerAngleLimit:upperAngleLimit);
        radarClip = round(radarClip, 3);

        %{
        imagesc(ang_grid(lowerAngleLimit:upperAngleLimit),rng_grid(1:rangeLimit),abs(resp));
        xlabel('Angle');
        ylabel('Range (m)');
        title('Range-Angle Map');
        %}

        radarClip = reshape(radarClip.', 1, []).';
        finalExport(CurrentFileNumber, currentFrameNumber+1) = mat2cell(radarClip, 5005);
        
    end


end

finalTable = cell2table(finalExport);
writetable(finalTable, "output.csv");

% EOF