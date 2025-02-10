function aggregator_matlab()
    % AGGREGATOR_MATLAB
    % Example aggregator script with extra checks and forced figure visibility.

    % Check for readNPY on path
    if ~exist('readNPY','file')
        error('readNPY.m not found on MATLAB path. Download it from MathWorks File Exchange or similar.');
    end

    % Prompt for CSV
    [csvName, csvPath] = uigetfile({'*.csv','CSV Files';'*.*','All Files'}, ...
        'Select event_log_runX.csv');
    if isequal(csvName,0)
        disp('No CSV selected. Exiting.');
        return;
    end
    csvFile = fullfile(csvPath, csvName);
    disp(['Chosen CSV: ', csvFile]);

    % Prompt for patches folder
    patchesDir = uigetdir('', 'Select patches_runX folder');
    if patchesDir==0
        disp('No patches folder. Exiting.');
        return;
    end
    disp(['Patches folder: ', patchesDir]);

    % aggregator dimension
    defDim   = '80';
    defNorm  = 'peak';
    prompt   = {'Aggregator dimension:', 'Normalization (none/peak/sum):'};
    answer   = inputdlg(prompt, 'Aggregator Settings', [1 40], {defDim, defNorm});
    if isempty(answer)
        disp('User cancelled aggregator settings. Exiting.');
        return;
    end
    aggregatorDim = str2double(answer{1});
    normalizationMode = lower(strtrim(answer{2}));
    if isnan(aggregatorDim) || aggregatorDim<1
        aggregatorDim= 80;
    end

    % Read CSV
    T = readtable(csvFile);
    nEvents = height(T);
    fprintf('CSV has %d rows (events).\n', nEvents);
    if nEvents<1
        disp('No events in CSV. Exiting.');
        return;
    end

    sumArray = zeros(aggregatorDim, aggregatorDim, 'single');
    validCount = 0;

    for i=1:nEvents
        patchFname = T.patch_filename{i};
        oldPatchPath= fullfile(patchesDir, patchFname);
        if ~isfile(oldPatchPath)
            fprintf('Event %d: patch not found: %s\n', i, oldPatchPath);
            continue;
        end

        patchData = readNPY(oldPatchPath);
        patchData = single(patchData);
        if isempty(patchData)
            fprintf('Event %d: patchData is empty.\n', i);
            continue;
        end

        % normalization
        switch normalizationMode
            case 'none'
                % do nothing
            case 'peak'
                mx = max(patchData(:));
                if mx>0
                    patchData= patchData/mx;
                end
            case 'sum'
                s = sum(patchData(:));
                if s>0
                    patchData= patchData/s;
                end
        end

        % align
        aligned = alignByPeak(patchData, aggregatorDim);
        if isempty(aligned)
            fprintf('Event %d: does not fit aggregatorDim\n', i);
            continue;
        end

        sumArray = sumArray + aligned;
        validCount= validCount+1;
    end

    fprintf('Valid events processed: %d\n', validCount);

    % We won't exit if validCount=0 -> we'll just produce an all-zero aggregator
    avgArray= sumArray / max(validCount,1); 

    % 2D Heatmap
    fig1= figure('Name','Aggregator 2D Heatmap','Visible','on');
    imagesc(avgArray); 
    axis image; 
    colormap('gray');
    colorbar; 
    title(sprintf('Aggregator (Peak Align) with %d valid patches', validCount));
    % optionally save:
    % saveas(fig1, 'aggregator_2d.png');

    % 3D Surface
    fig2= figure('Name','Aggregator 3D Surface','Visible','on');
    surf(double(avgArray),'EdgeColor','none');
    colormap('jet'); 
    colorbar; 
    title('3D Surface'); 
    view(45,60);

    % If you want a distribution of sum_intensity:
    if ismember('sum_intensity', T.Properties.VariableNames)
        figure('Name','Sum Int Dist','Visible','on');
        histogram(T.sum_intensity,20);
        xlabel('sum_intensity');
        ylabel('Count');
        title('Distribution of sum_intensity');
    end

    disp('All figures displayed. Close them to end.');

end

function aligned = alignByPeak(patchData, aggregatorDim)
    % find brightest
    [mx, idx] = max(patchData(:));
    if mx<=0
        aligned=[];
        return;
    end
    [pr, pc] = ind2sub(size(patchData), idx);
    [ph, pw]= size(patchData);

    centerR = floor(aggregatorDim/2);
    centerC = floor(aggregatorDim/2);

    shiftR= centerR - (pr-1);
    shiftC= centerC - (pc-1);

    startR= shiftR+1;
    startC= shiftC+1;
    endR = startR+ ph-1;
    endC = startC+ pw-1;

    if startR<1 || startC<1 || endR>aggregatorDim || endC>aggregatorDim
        aligned=[];
        return;
    end

    aligned= zeros(aggregatorDim, aggregatorDim, 'single');
    aligned(startR:endR, startC:endC)= patchData;
end
