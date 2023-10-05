%%% This script is used to translate the 3x1 radar data array into a
%%% Frames x Chirps x Antennas x Samples radar cube array.
%%% Each element in the result array will be a single frame as a radar
%%% cube, in format Chirps X Antennas X Samples

function[retVal] = shuffleRxSorted(data, frames, receive, chirps, samples)

    % Init return array
    retVal = zeros(frames, chirps, receive, samples);

    % loop through the different dimensions of the array
    for r=1:receive
        for f=1:frames
            for c=1:chirps
                for s=1:samples

                    % Select the right value from the source data by moving
                    % along in 'blocks'
                    selector = ((f-1) * (chirps + samples)) + ((c-1) * (samples)) + s;
                    retVal(f, c, r, s) = data(r, selector);

                end
            end
        end
    end

    % End of function
end

