function [bestEpsilon, bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    predictions = pval < epsilon;       %Find predictions based on threshold epsilon
    TP = sum(yval .* predictions);      %True Positive (Examples predicted as 1 and it's actually 1)
    TotalP = sum(predictions == 1);     %Total examples predicted as 1
    prec = TP/ TotalP;                  %Calculate Precision
    TotalAP = sum(yval == 1);           %Total examples that's actually 1
    recall = TP/TotalAP;                %Calculate Recall
    F1 = 2*prec*recall / (prec+recall); %Calculate F1 Score
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
