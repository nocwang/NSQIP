function [pred_err, opt_variables, ROC] = RBFSVM_withParaTuning(xtrain, ytrain, xtest, ytest,Cs,Sigmas,Options)
% % function [pred_err, opt_variables, ROC] = RBFSVM_withParaTuning(allData,Cs,Sigmas,Options)
% % linear svm with parameter C tuning. By default, the C is automatically
% % scaled by the sample size of each class.
% % Note: to get meaningful ROC=[false_alarm, detection], the labels needs to be 1 (normal) and 2
% % (abnormal)
% % Options: additional Options that gives the freedom of further tuning the
% %   function.
% % Options.Flag_AUC: 1 (meaning the pred_err is AUC), 0 (pred_err is just
% %   overall error rate)

% xtrain = allData.xtrain;
% ytrain = allData.ytrain;
% xtest = allData.xtest;
% ytest = allData.ytest;

% random permutation in case the crossval function of matlab doesn't do it
randIdx = randperm(length(ytrain));
xtrain = xtrain(randIdx,:);
ytrain = ytrain(randIdx);

% finding the optimal parameters by cross validation
val_err = zeros(length(Cs), length(Sigmas)); % record average validation error
for i_c = 1:length(Cs)
    t_C = Cs(i_c);
    for i_sig = 1:length(Sigmas)
        t_Sigma = Sigmas(i_sig);
%         if exist('Options','var')
            fun = @(xtrain, ytrain, xtest, ytest)(RBFSVM_withCandSigma(xtrain, ytrain, xtest, ytest, t_C, t_Sigma,0, Options));
%         else
%             fun = @(xtrain, ytrain, xtest, ytest)(RBFSVM_withCandSigma(xtrain, ytrain, xtest, ytest, t_C, t_Sigma,0));
%         end            
        vals = crossval(fun, xtrain, ytrain, 'Kfold',3);
        val_err(i_c, i_sig) = mean(vals);
    end
end

% if exist('Options','var') && isfield(Options,'Flag_AUC') && (Options.Flag_AUC == 1)
    [~, optInd] = max(val_err(:));
% else    
%     [~, optInd] = min(val_err(:));
% end
[optC_Ind, optSigma_Ind] = ind2sub(size(val_err),optInd);
optC = Cs(optC_Ind);
optSigma = Sigmas(optSigma_Ind);

% the autoscale parameter is always 0 and if we would like to normalized
% the input data, we do it manually.
svmStruct = svmtrain(xtrain, ytrain, 'Kernel_Function','rbf','RBF_SIGMA',optSigma,'boxconstraint',optC,'options',statset('MaxIter',100000),'autoscale',0, 'kernelcachelimit', 100000);
% ypred = svmclassify(svmStruct, xtest);
% pred_err = sum(ypred(:)~=ytest(:))/length(ytest);
% % ROC, more than just one pred_err
sv = svmStruct.SupportVectors;
alphaHat = svmStruct.Alpha; % the matlab convention makes alphaHat combining alpha and y
bias = svmStruct.Bias;
kfun = svmStruct.KernelFunction;
kfunargs = svmStruct.KernelFunctionArgs;
pred_vals = (feval(kfun,sv,xtest, kfunargs{:})'*alphaHat(:))+bias;
% thresholds = sort(pred_vals,1,'ascend');
% miss = zeros(length(thresholds),1);
% false_alarm = zeros(length(thresholds),1);
% for i_thr = 1:length(thresholds)
%     ypred = (pred_vals > thresholds(i_thr))+1;
%     ypred = 3-ypred;
%     miss(i_thr) = sum(ypred == 1 & ytest == 2)/sum(ytest == 2);
%     false_alarm(i_thr) = sum(ypred == 2 & ytest == 1)/sum(ytest == 1);
% end
% ROC = [false_alarm, 1-miss];
% 
% 
% if exist('Options','var') && isfield(Options,'Flag_AUC') && (Options.Flag_AUC == 1)
%     pred_err = calcAUCgivenROC(ROC);
% end
     [ROC(:,1),ROC(:,2),~,pred_err]=perfcurve(ytest,-1*pred_vals,2);%ROC.thresholds  here it's faster!!! so replace wuyang's code
     opt_variables.optC = optC;
     opt_variables.optSigma = optSigma;
     opt_variables.svmStruct = svmStruct;
     opt_variables.ROC = ROC;
