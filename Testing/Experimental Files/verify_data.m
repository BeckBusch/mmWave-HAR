%
% Introduction
%     This verification script takes radarCube mat file as input. It
% randomly selects one chirp data to search for the peak. The peak
% information is printed for user for verify the object.
%
%
% Requirement for the test
%     The captured bin files should be generated from mmwave studio with a
% corner reflector at a certain range. Run post_processing script to parse
% the JSON file and bin file(s) to generate radarCube mat file for this
% test.
%
% Syntax:
%    verify_data('radarCubeMatFileName')
%
function verify_data(radarCubeMatFile)

    try
        % Load radar cube data
        load(radarCubeMatFile);
    catch
        error("Error reading input *.mat file");
    end

    % Get radar cube params
    radarCubeParams = radarCube.dim;
    rfParams = radarCube.rfParams;

    % Radom select frame/chirp/rxChan
    frameIdx = ceil(radarCubeParams.numFrames * (rand));
    chirpIdx = ceil(radarCubeParams.numChirps * (rand));
    rxChan = ceil(radarCubeParams.numRxChan * (rand));

    fprintf('radarCube data verification after processing the captured bin file...\n');
    fprintf('-------------------------------------------------------------------\n');
    fprintf('Input radar cube File contains %d frames, %d chirps per frame, %d Rx Channel of data.\n', ...
                                radarCubeParams.numFrames, ...
                                radarCubeParams.numChirps, ...
                                radarCubeParams.numRxChan);
    fprintf('Test is performed on a random selected chirp.\n');
    fprintf('\tframe= %d, \tchirp=%d, \trxChan=%d\n',...
                                frameIdx, chirpIdx, rxChan);

    % Load radarCube data
    frameData = radarCube.data{frameIdx};
    rangeData(:) = frameData(chirpIdx,rxChan,:);
    rangeProfile = abs(rangeData);

    % Debug Plot
    %figure(1);
    %plot(abs(rangeData));

    [peakVal, peakIdx] = max(rangeProfile);

    fprintf('Result: \n \tpeak range index = %d\n \tpeakVal = %f\n \tpeak range = %f(meter)\n', ...
                                peakIdx, round(peakVal,2), ...
                                peakIdx*rfParams.rangeResolutionsInMeters);

end
