function [ res, prec, rec, acc ] = Evaluate( gtLabelID, predLabelID, fps )
%EVALUATE
% A function to evaluate the performance of the phase recognition method
% providing jaccard index, precision, and recall for each phase 
% and accuracy over the surgery. All metrics are computed in a relaxed
% boundary mode.
% OUTPUT:
%    res: the jaccard index per phase (relaxed) - NaN for non existing phase in GT
%    prec: precision per phase (relaxed)        - NaN for non existing phase in GT
%    rec: recall per phase (relaxed)            - NaN for non existing phase in GT
%    acc: the accuracy over the video (relaxed)

oriT = 10 * fps; % 10 seconds relaxed boundary

res = []; prec = []; rec = [];
diff = predLabelID - gtLabelID;
updatedDiff = [];

% obtain the true positive with relaxed boundary
for iPhase = 1:8 % nPhase
    gtConn = bwconncomp(gtLabelID == iPhase);
    
    for iConn = 1:gtConn.NumObjects
        startIdx = min(gtConn.PixelIdxList{iConn});
        endIdx = max(gtConn.PixelIdxList{iConn});

        curDiff = diff(startIdx:endIdx);

        % in the case where the phase is shorter than the relaxed boundary
        t = oriT;
        if(t > length(curDiff))
            t = length(curDiff);
            disp(['Very short phase ' num2str(iPhase)]);
        end
        
        % relaxed boundary
        if(iPhase == 5 || iPhase == 6) % Gallbladder dissection and packaging might jump between two phases
            curDiff(curDiff(1:t)==-1) = 0; % late transition
            curDiff(curDiff(end-t+1:end)==1 | curDiff(end-t+1:end)==2) = 0; % early transition
        elseif(iPhase == 7 || iPhase == 8) % Gallbladder dissection might jump between two phases
            curDiff(curDiff(1:t)==-1 | curDiff(1:t)==-2) = 0; % late transition
            curDiff(curDiff(end-t+1:end)==1 | curDiff(end-t+1:end)==2) = 0; % early transition
        else
            % general situation
            curDiff(curDiff(1:t)==-1) = 0; % late transition
            curDiff(curDiff(end-t+1:end)==1) = 0; % early transition
        end
        a = sum(curDiff==0);
        updatedDiff(startIdx:endIdx) = curDiff;
    end
end

% compute jaccard index, prec, and rec per phase
for iPhase = 1:8
    gtConn = bwconncomp(gtLabelID == iPhase);
    predConn = bwconncomp(predLabelID == iPhase);
    
    if(gtConn.NumObjects == 0)
        % no iPhase in current ground truth, assigned NaN values
        % SHOULD be excluded in the computation of mean (use nanmean)
        res(end+1) = NaN;
        prec(end+1) = NaN;
        rec(end+1) = NaN;
        continue;
    end
    
    iPUnion = union(vertcat(predConn.PixelIdxList{:}), vertcat(gtConn.PixelIdxList{:}));
    tp = sum(updatedDiff(iPUnion) == 0);
    jaccard = tp/length(iPUnion);
    jaccard = jaccard * 100;

%     res(end+1, :) = [iPhase jaccard];
    res(end+1) = jaccard;

    % Compute prec and rec
    indx = (gtLabelID == iPhase);

    sumTP = tp; % sum(predLabelID(indx) == iPhase);
    sumPred = sum(predLabelID == iPhase);
    sumGT = sum(indx);
    
    if sumPred == 0
        prec(end+1) = 0;
    else
        prec(end+1) = sumTP * 100 / sumPred;
    end
    %prec(end+1) = sumTP * 100 / sumPred;   %%%changed by ymjin
    rec(end+1)  = sumTP * 100 / sumGT;    
end

% compute accuracy
accnum = sum(updatedDiff==0);
acc = accnum / length(gtLabelID);
acc = acc * 100;
fprintf('accuracy = %f, right number =%d\n', acc, accnum);

end

