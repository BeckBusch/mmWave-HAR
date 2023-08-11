FrameCount = 25 * 6;
ChirpCount = 128;
RxAntennaCount = 4;
AdcSampleCount = 256;

BinaryFilePath = "RadarBins/boxing_6s.bin";

% Receive, (frame, chirp, sample) - postproc format
rxData = readDCA1000(BinaryFilePath, FrameCount, ChirpCount, RxAntennaCount, AdcSampleCount);
% Frame, chirp, receive, sample - usual format
ArrangedData = shuffleRxSorted(rxData, FrameCount, RxAntennaCount, ChirpCount, AdcSampleCount);
% Frame, receive, chirp, sample - matlab radar format
ArrangedData = permute(ArrangedData, [1 3 2 4]);    

