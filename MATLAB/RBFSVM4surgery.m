%RBF SVM with RBFSVM_withParaTuning & svmtrain in MATLAB
clear; clc;
load('data4SVM_bf2011.mat','random10times','data','name','y')
% needdelete=find(ismember(name,'"returnor"'))%[];
% data(:,needdelete)=[];%
% name(needdelete)=[];%

y(y==0)=-1;
TrainingRatio = 0.80;
N = 10;
N_fold = 3;

removed_dim10= cell(N,1);
RBFSVM_results = zeros(N,1);
RBFSVM_resultsVars = cell(N,1);
RBFSVM_ROC= cell(N,1);

% % parameter settings
Cs = [ 10, 1, 0.1];% % [ 10, 1, 0.1, 0.01, 0.001];%  [100,10, 1, 0.1, 0.01];%
Sigmas = [100, 10, 1]; %[1000, 100, 10, 1]; % [10, 3, 1, 0.3];% Sigmas = [10]; is best for RBF(end)
basic_Options.Flag_AUC = 1;
basic_Options.Flag_No_C_Rescaling = 0;

tic;
time1 = cputime;
for k = 1:N
    %% Data Preprocessing;
    %labels=2*(y==1)+(1)*(y==0);
    allFeas=data;
    
    randPermIndex =random10times(k,:);%randPermIndex = randperm(size(allFeas,1));
    index_selected_train = randPermIndex(1:floor(size(allFeas,1)*TrainingRatio));
    index_selected_test = randPermIndex(1+floor(size(allFeas,1)*TrainingRatio):end);
    
    % form training data and test data
    allTrainData = allFeas(index_selected_train,:); % the whole set of samples for training but some of them may not be used.
    allTrainLabel = y(index_selected_train);
    
    index_train_class2 = find(allTrainLabel == 1);%abnormal
    index_train_class1 = find(allTrainLabel == -1);
    % balance the two classes for training
    xtrain = allTrainData([index_train_class2;index_train_class1(1:length(index_train_class2))],:);
    ytrain = allTrainLabel([index_train_class2;index_train_class1(1:length(index_train_class2))]);
    xtest = allFeas(index_selected_test,:);
    ytest = y(index_selected_test);
    % IMPORTANT: scaling the input features
    removed_dim=find(std(xtrain)< 1e-3);
    removed_dim10{k} =removed_dim;
    xtrain(:,removed_dim) = [];
    xtest(:,removed_dim) = [];
    t_mode = mode(xtrain);
    xtrain = xtrain - repmat(t_mode, size(xtrain,1),1);
    xtest = xtest - repmat(t_mode, size(xtest,1),1);
    %std_xtrain=range(xtrain);
    std_xtrain=std(xtrain);
    std_xtrain(std_xtrain==0)=1;
    xtrain=xtrain./repmat(std_xtrain, size(xtrain,1),1);
    xtest=xtest./repmat(std_xtrain, size(xtest,1),1);
    
    %%
    ytrain = (ytrain+3)/2; % convention of SVM requires label 1 and 2
    ytest = (ytest+3)/2;
    [RBFSVM_results(k), RBFSVM_resultsVars{k}, RBFSVM_ROC{k}] = RBFSVM_withParaTuning(xtrain, ytrain, xtest, ytest,Cs,Sigmas,basic_Options);
    RBFSVM_results(k)
end
toc;
time4 = cputime-time1;

meanAUC_RBF=  mean(RBFSVM_results)
stdAUC_RBF=  std(RBFSVM_results)
fname = sprintf('../NSQIP0513/result_bf2011_20161231_%d_percent_RBFSVM_wynorm_Cs5.mat',TrainingRatio*100);
save(fname);

for k=1:10
    [RBFSVM_resultsVars{k, 1}.optSigma,RBFSVM_resultsVars{k, 1}.optC]
end