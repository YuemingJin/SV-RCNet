close all; clear all;

phaseGroundTruths = {'test_workflow_video_01.txt','test_workflow_video_02.txt','test_workflow_video_03.txt',...
    'test_workflow_video_04.txt','test_workflow_video_05.txt','test_workflow_video_06.txt',...
    'test_workflow_video_07.txt','test_workflow_video_08.txt','test_workflow_video_09.txt',...
    'test_workflow_video_10.txt','test_workflow_video_11.txt','test_workflow_video_12.txt',...
    'test_workflow_video_13.txt','test_workflow_video_14.txt'};

phases = {'TrocarPlacement', 'Preparation',  'CalotTriangleDissection', ...
    'ClippingCutting', 'GallbladderDissection',  'GallbladderPackaging', 'CleaningCoagulation', ...
    'GallbladderRetraction'};

fps = 25;

for i = 1:14
    %length(phaseGroundTruths)
    
    gt_rootfolder = 'path/to/ground/truth/folder/';
    phaseGroundTruth_single = phaseGroundTruths{i};
    phaseGroundTruth = [gt_rootfolder phaseGroundTruth_single(1:end-4) '.txt'];
    
    model_rootfolder = 'path/to/model/folder/';
    predFile = [model_rootfolder phaseGroundTruth_single(1:end-4) 'name.txt'];
    
    [gt] = ReadPhaseLabel(phaseGroundTruth);
    [pred] = ReadPhaseLabel(predFile);
    
    if(size(gt{1}, 1) ~= size(pred{1},1) || size(gt{2}, 1) ~= size(pred{2},1))
        error(['ERROR:' ground_truth_file '\nGround truth and prediction have different sizes']);
    end
    
    if(~isempty(find(gt{1} ~= pred{1})))
        error(['ERROR: ' ground_truth_file '\nThe frame index in ground truth and prediction is not equal']);
    end
    
    % reassigning the phase labels to numbers
    gtLabelID = [];
    predLabelID = [];
    for j = 1:length(phases)
        gtLabelID(find(strcmp(phases{j}, gt{2}))) = j;
        predLabelID(find(strcmp(phases{j}, pred{2}))) = j;
    end
    
    % compute jaccard index, precision, recall, and the accuracy
    [jaccard(:,i), prec(:,i), rec(:,i), acc(i)] = Evaluate(gtLabelID, predLabelID, fps);
    
end

% Compute means and stds
index = find(jaccard>100);
jaccard(index)=100;
meanJaccPerPhase = nanmean(jaccard, 2);
meanJacc = mean(meanJaccPerPhase);
stdJacc = std(meanJaccPerPhase);
for h = 1:8
    jaccphase = jaccard(h,:);
    meanjaccphase(h) = nanmean(jaccphase);
    stdjaccphase(h) = nanstd(jaccphase);
end

index = find(prec>100);
prec(index)=100;
meanPrecPerPhase = nanmean(prec, 2);
meanPrec = nanmean(meanPrecPerPhase);
stdPrec = nanstd(meanPrecPerPhase);
for h = 1:8
    precphase = prec(h,:);
    meanprecphase(h) = nanmean(precphase);
    stdprecphase(h) = nanstd(precphase);
end

index = find(rec>100);
rec(index)=100;
meanRecPerPhase = nanmean(rec, 2);
meanRec = mean(meanRecPerPhase);
stdRec = std(meanRecPerPhase);
for h = 1:8
    recphase = rec(h,:);
    meanrecphase(h) = nanmean(recphase);
    stdrecphase(h) = nanstd(recphase);
end


meanAcc = mean(acc);
stdAcc = std(acc);

% Display results
fprintf('model is :%s\n', model_rootfolder);
disp('================================================');
disp([sprintf('%25s', 'Phase') '|' sprintf('%6s', 'Jacc') '|'...
    sprintf('%6s', 'Prec') '|' sprintf('%6s', 'Rec') '|']);
disp('================================================');
for iPhase = 1:length(phases)
    disp([sprintf('%25s', phases{iPhase}) '|' sprintf('%6.2f', meanJaccPerPhase(iPhase)) '|' ...
        sprintf('%6.2f', meanPrecPerPhase(iPhase)) '|' sprintf('%6.2f', meanRecPerPhase(iPhase)) '|']);
    disp('---------------------------------------------');
end
disp('================================================');

disp(['Mean jaccard: ' sprintf('%5.2f', meanJacc) ' +- ' sprintf('%5.2f', stdJacc)]);
disp(['Mean accuracy: ' sprintf('%5.2f', meanAcc) ' +- ' sprintf('%5.2f', stdAcc)]);
disp(['Mean precision: ' sprintf('%5.2f', meanPrec) ' +- ' sprintf('%5.2f', stdPrec)]);
disp(['Mean recall: ' sprintf('%5.2f', meanRec) ' +- ' sprintf('%5.2f', stdRec)]);
