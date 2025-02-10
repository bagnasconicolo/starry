function aggregator_matlab()
% AGGREGATOR_MATLAB
% Example MATLAB script to average event patches (with margin) and do peak alignment.
%
% Steps:
%   1) Asks user for CSV + patches folder.
%   2) Asks user for aggregatorDim, normalizationMode, etc.
%   3) Loops over each CSV row:
%       - read the .npy patch (via readNPY).
%       - optional normalization (peak or sum).
%       - align by peak => aggregator center
%       - sum into sum_array
%   4) average => sum_array / nEvents
%   5) simple plots (imagesc, surf, distribution).
%
% NOTE: requires a "readNPY.m" function to load .npy files in MATLAB.
% see: https://www.mathworks.com/matlabcentral/fileexchange/75166-npy-matlab
%
% Usage:
%   aggregator_matlab
%   - follow prompts, see plots.

    %% 1) Prompt user for CSV + patches folder
    [csvName, csvPath] = uigetfile({'*.csv','CSV Files';'*.*','All Files'}, ...
        'Select event_log_runX.csv');
    if isequal(csvName,0)
        disp('No CSV selected. Exiting.');
        return;
    end
    csvFile = fullfile(csvPath, csvName);

    patchesDir = uigetdir('', 'Select patches_runX folder');
    if patchesDir == 0
        disp('No patches folder selected. Exiting.');
        return;
    end

    %% 2) aggregator dimension
    prompt = {'Enter aggregator dimension (e.g. 80):','Normalization Mode (none/peak/sum):'};
    dlgtitle = 'Aggregator Settings';
    definput = {'80','peak'};
    answer = inputdlg(prompt, dlgtitle, [1 60], definput);
    if isempty(answer)
        disp('No aggregator settings provided. Exiting.');
        return;
    end
    aggregatorDim = str2double(answer{1});
    normalizationMode = lower(strtrim(answer{2}));

    if isnan(aggregatorDim) || aggregatorDim<1
        aggregatorDim = 80;
    end

    %% 3) read CSV as table
    T = readtable(csvFile);
    nEvents = height(T);
    if nEvents<1
        disp('No events in CSV. Exiting.');
        return;
    end
    fprintf('Loaded %d events from %s\n', nEvents, csvFile);

    %% 4) Prepare aggregator
    sumArray = zeros(aggregatorDim, aggregatorDim, 'single');
    % optional for variance or max projection, do:
    % sqSumArray = zeros(aggregatorDim, aggregatorDim, 'single');
    % maxArray   = zeros(aggregatorDim, aggregatorDim, 'single');

    validCount = 0;

    %% 5) loop over each event
    for i = 1:nEvents
        % get patch filename from the table
        patchFname = T.patch_filename{i};
        if ~ischar(patchFname)
            continue;
        end
        oldPatchPath = fullfile(patchesDir, patchFname);
        if ~exist(oldPatchPath,'file')
            fprintf('Patch not found: %s\n', oldPatchPath);
            continue;
        end

        % load .npy
        patchData = readNPY(oldPatchPath);  % requires external readNPY.m
        patchData = single(patchData);

        % optional normalization
        switch normalizationMode
            case 'none'
                % do nothing
            case 'peak'
                mx = max(patchData(:));
                if mx>0
                    patchData = patchData / mx;
                end
            case 'sum'
                s = sum(patchData(:));
                if s>0
                    patchData = patchData / s;
                end
            otherwise
                % fallback none
        end

        % align by peak
        aligned = alignByPeak(patchData, aggregatorDim);
        if isempty(aligned)
            % doesn't fit aggregator
            continue;
        end

        % accumulate
        sumArray = sumArray + aligned;

        validCount = validCount + 1;
    end

    if validCount<1
        disp('No valid patches after alignment. Exiting.');
        return;
    end

    %% 6) average aggregator
    avgArray = sumArray ./ single(validCount);

    %% 7) simple visualization
    figure('Name','Aggregator 2D Heatmap');
    imagesc(avgArray);
    colormap('gray');
    colorbar;
    axis image;
    title(sprintf('Avg Aggregator (Peak-Aligned) - %d events', validCount));

    figure('Name','Aggregator 3D Surface');
    surf(double(avgArray),'EdgeColor','none');
    colormap('jet');
    colorbar;
    title('3D Surface of Aggregator');
    xlabel('X'); ylabel('Y'); zlabel('Intensity');
    view(45,65);

    %% 8) distribution example: sum_intensity
    % If you want to see the distribution of sum_intensity from T:
    if ismember('sum_intensity', T.Properties.VariableNames)
        sumInt = T.sum_intensity;
        figure('Name','Sum Intensity Distribution');
        histogram(sumInt,20);
        xlabel('sum_intensity');
        ylabel('Count');
        title('Distribution of sum_intensity');
    end

    disp('Done. Close figures to end.');

end

%% Helper function: Align patch by peak
function aligned = alignByPeak(patchData, aggregatorDim)
    % find brightest pixel
    [mx, idx] = max(patchData(:));
    if mx<=0
        aligned = [];
        return;
    end
    [pr, pc] = ind2sub(size(patchData), idx);

    ph = size(patchData,1);
    pw = size(patchData,2);

    centerR = floor(aggregatorDim/2);
    centerC = floor(aggregatorDim/2);

    shiftR = centerR - (pr-1);  % -1 because indexing difference
    shiftC = centerC - (pc-1);

    startR = shiftR + 1;  % shifting to 1-based index
    startC = shiftC + 1;
    endR   = startR + ph - 1;
    endC   = startC + pw - 1;

    if startR<1 || startC<1 || endR>aggregatorDim || endC>aggregatorDim
        aligned = [];
        return;
    end

    aligned = zeros(aggregatorDim, aggregatorDim, 'single');
    aligned(startR:endR, startC:endC) = patchData;
end
