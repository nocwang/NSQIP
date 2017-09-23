%% LIBLINEAR 2LR 2SVM methods for surgery
% based on function file plotroc_liblinear 
% first install liblinear from https://www.csie.ntu.edu.tw/~cjlin/liblinear/
clear; clc;
% % which predict
% % /share/pkg/liblinear/2.1/install/lib/predict.mexa64
% % which plotroc_liblinear

load('data4SVM_all_133_v7.mat','data','random10times','y')
load('name_new133.mat')

%% Data preprocessing 1: delete highly related features
% %  drop ** which is highly related  Number of to **
%load('data4SVM_all_151.mat','random10times','data','name','y')
% N_NAME={'"nneurodef"','"ncnscoma"','"npulembol"','"noprenafl"','"nwndinfd"','"ndehis"','"ncnscva"','"noupneumo"','"nothdvt"','"nfailwean"','"nsupinfec"','"nothseshock"','"nurninfec"','"nothsysep"','"nrenainsf"','"norgspcssi"','"ncdmi"','"nothbleed"','"nothgrafl"','"ncdarrest"','"nreintub"' }
% N_ID=find(ismember(name,N_NAME))
% %name(N_ID)
% for m=1:length(N_NAME)
%     str=N_NAME{m};%'"npulembol"'
%     N_NAME{m}=str([1,3:end])%str(1),str(3:end)
% end
% needdelete=find(ismember(name,N_NAME))
% data(:,needdelete)=[];%
% name(needdelete)=[];%
%% parameter settings
y(y==0)=-1;
TrainingRatio =0.4;
N = 10;
N_fold = 3;

removed_dim10= cell(N,1);
L2LR_AUC=zeros(N,1);
L2LR_BESTC=zeros(N,1);
L2LR_model_liblinear= cell(N,1);
L1LR_AUC=zeros(N,1);
L1LR_BESTC=zeros(N,1);
L1LR_model_liblinear= cell(N,1);
L2L1_AUC=zeros(N,1);
L2L1_BESTC=zeros(N,1);
L2L1_model_liblinear= cell(N,1);
L1L2_AUC=zeros(N,1);
L1L2_BESTC=zeros(N,1);
L1L2_model_liblinear= cell(N,1);
L2LR_ROC = cell(N,1);
% L1LR_ROC = cell(N,1);
% L2L1_ROC = cell(N,1);
% L1L2_ROC = cell(N,1);

tic;
time1 = cputime;
parfor k = 1:N%for k = 1:N
    %% Data Preprocessing;
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
    removed_dim=find(std(xtrain)< 1e-6);
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
    
    %% tuning Parameters by CV
    L2LR__bestcv = 0;  L1LR_bestcv = 0;    LSVM_bestcv = 0; L2L1_bestcv =0;L1L2_bestcv=0;
    for  log10c= -2:0,% -3:3
        %0 -- L2-regularized logistic regression (primal)  %cmd_method=['-s 0'];
        cmd = ['-B -q -s 0 -v 3 -c ', num2str(10^log10c)];%cmd = ['-s 0 -v 3 -c ', num2str(10^log10c)];
        L2LR_cv=plotroc_liblinear(ytrain, sparse(xtrain), cmd);
        if (L2LR_cv >= L2LR__bestcv)
            L2LR__bestcv = L2LR_cv; L2LR_BESTC(k) = 10^log10c;
        end
        %6 -- L1-regularized logistic regression           %cmd_method=['-s 6'];
        cmd = ['-B -q -s 6 -v 3 -c ', num2str(10^log10c)];%cmd = ['-s 6 -v 3 -c ', num2str(10^log10c)];
        L1LR_cv=plotroc_liblinear(ytrain, sparse(xtrain), cmd);
        if (L1LR_cv >= L1LR_bestcv)
            L1LR_bestcv = L1LR_cv; L1LR_BESTC(k) = 10^log10c;
        end
        %5 -- L1-regularized L2-loss support vector classification
        cmd = ['-B -q -s 5 -v 3 -c ', num2str(10^log10c)];
        L1L2_cv=plotroc_liblinear(ytrain, sparse(xtrain), cmd);
        if (L1L2_cv >= L1L2_bestcv)
            L1L2_bestcv = L1L2_cv; L1L2_BESTC(k) = 10^log10c;
        end
    end
    for  log10c= -4:-2,
        %3 -- L2-regularized L1-loss support vector classification (dual)
        %%cmd_method=['-s 3'];  L2L1_BESTC==1e-4 since it's dual form
        cmd = ['-B -q -s 3 -v 3 -c ', num2str(10^log10c)];%cmd = ['-s 0 -v 3 -c ', num2str(10^log10c)];
        L2L1_cv=plotroc_liblinear(ytrain, sparse(xtrain), cmd);
        if (L2L1_cv >= L2L1_bestcv)
            L2L1_bestcv = L2L1_cv; L2L1_BESTC(k) = 10^log10c;
        end
    end
    %[L2LR_BESTC(k), L1LR_BESTC(k), L2L1_BESTC(k),L1L2_BESTC(k)]
    L2LR_model_liblinear{k} = train(ytrain, sparse(xtrain),['-B -q -s 0 -c ', num2str(L2LR_BESTC(k))]);%
    L2LR_AUC(k)=plotroc_liblinear(ytest,sparse(xtest), L2LR_model_liblinear{k});
    %L2LR_AUC_tr(k)=plotroc_liblinear(ytrain,sparse(xtrain), L2LR_model_liblinear{k});
    pred_vals =xtest*L2LR_model_liblinear{k}.w(1:end-1)'+L2LR_model_liblinear{k}.bias;
    [L2LR_ROC{k}.test(:,1),L2LR_ROC{k}.test(:,2),L2LR_ROC{k}.thresholds,L2LR_ROC{k}.AUC_test]=perfcurve(ytest,pred_vals,1);
    
    L1LR_model_liblinear{k} = train(ytrain, sparse(xtrain),['-B -q -s 6 -c ', num2str(L1LR_BESTC(k))]);%
    L1LR_AUC(k)=plotroc_liblinear(ytest,sparse(xtest), L1LR_model_liblinear{k});
    %plotroc_liblinear(ytrain,sparse(xtrain), L1LR_model_liblinear{k});
    L2L1_model_liblinear{k} = train(ytrain, sparse(xtrain),['-B -q -s 3 -c ', num2str(L2L1_BESTC(k))]);%
    L2L1_AUC(k)=plotroc_liblinear(ytest,sparse(xtest), L2L1_model_liblinear{k});
    %plotroc_liblinear(ytrain,sparse(xtrain), L2L1_model_liblinear{k})
    L1L2_model_liblinear{k} = train(ytrain, sparse(xtrain),['-B -q -s 5 -c ', num2str(L1L2_BESTC(k))]);%
    L1L2_AUC(k)=plotroc_liblinear(ytest,sparse(xtest), L1L2_model_liblinear{k});
end
toc;
time4 = cputime-time1

meanAUC_L2LR=mean(L2LR_AUC)
stdAUC_L2LR_=std(L2LR_AUC)
meanAUC_L1LR=mean(L1LR_AUC)
stdAUC_L1LR_=std(L1LR_AUC)
meanAUC_L2L1=mean(L2L1_AUC)
stdAUC_L2L1_=std(L2L1_AUC)
meanAUC_L1L2=mean(L1L2_AUC)
stdAUC_L1L2_=std(L1L2_AUC)


[L2LR_BESTC, L1LR_BESTC, L2L1_BESTC,L1L2_BESTC]
fname = sprintf('result%d_all_%d_percent_2SVMLR_wynorm_Cs_bias.mat',size(data,2),TrainingRatio*100)%_6feature
save(fname);
%% top 10
L2LR_IMscore=zeros(size(data,2),10);id_top10=zeros(10,10);L2LR_IMrank=zeros(size(data,2),10);
for k=1:10
    L2LR_IMscore(setdiff(1:size(data,2),removed_dim10{k}),k)=L2LR_model_liblinear{k, 1}.w(1:end-1);%/abs(L2LR_model_liblinear{k, 1}.bias);
    [B,I] = sort(abs(L2LR_IMscore(:,k)),'descend');
    L2LR_IMrank(I,k)=1:size(data,2);
    IMscore_id10=I(1:10);
    id_top10(:,k)=IMscore_id10;
end
L2LR_median_IMrank=median(L2LR_IMrank,2);
[L2LR_B_median_IMrank,L2LR_I_median_IMrank] = sort(L2LR_median_IMrank,'ascend');

% [ num2cell( L2LR_I_median_IMrank(1:15)),name(L2LR_I_median_IMrank(1:15))'   num2cell(median(L2LR_IMscore(L2LR_I_median_IMrank(1:15),:),2))]
L2LR_summary= [ num2cell( L2LR_I_median_IMrank), name(L2LR_I_median_IMrank)', num2cell(median(L2LR_IMscore(L2LR_I_median_IMrank,:),2))];
% load('corr133.mat','r460','summary_corr')
% load('name_new133.mat')
% L2LR_summary= [name(L2LR_I_median_IMrank)'   num2cell(median(L2LR_IMscore(L2LR_I_median_IMrank,:),2))  num2cell(r460(L2LR_I_median_IMrank))];
%%
L1LR_IMscore=zeros(size(data,2),10);names=[];id_top10=zeros(10,10);L1LR_IMrank=zeros(size(data,2),10);
for k=1:10
    L1LR_IMscore(setdiff(1:size(data,2),removed_dim10{k}),k)=L1LR_model_liblinear{k, 1}.w/abs(L1LR_model_liblinear{k, 1}.bias);
    [B,I] = sort(abs(L1LR_IMscore(:,k)),'descend');
    L1LR_IMrank(I,k)=1:size(data,2);
    IMscore_id10=I(1:10);
    id_top10(:,k)=IMscore_id10;
end
L1LR_median_IMrank=median(L1LR_IMrank,2);
[L1LR_B_median_IMrank,L1LR_I_median_IMrank] = sort(L1LR_median_IMrank,'ascend');
%L1LR_summary= [name(L1LR_I_median_IMrank(1:15))'   num2cell(median(L1LR_IMscore(L1LR_I_median_IMrank(1:15),:),2))]
L1LR_summary= [name(L1LR_I_median_IMrank)'   num2cell(median(L1LR_IMscore(L1LR_I_median_IMrank,:),2))];
%%
L2L1SVM_IMscore=zeros(size(data,2),10);id_top10=zeros(10,10);L2L1SVM_IMrank=zeros(size(data,2),10);%names=[];
for k=1:10
    L2L1SVM_IMscore(setdiff(1:size(data,2),removed_dim10{k}),k)=L2L1_model_liblinear{k, 1}.w/abs(L2L1_model_liblinear{k, 1}.bias);
    %SPLinSVM_resultsVars{k,1}.opt_beta/abs(SPLinSVM_resultsVars{k, 1}.opt_beta0);
    [B,I] = sort(abs(L2L1SVM_IMscore(:,k)),'descend');
    L2L1SVM_IMrank(I,k)=1:size(data,2);
    %IMscore_id10=find(abs(IMscore)>0.01);
    IMscore_id10=I(1:10);%B(1:15)
    id_top10(:,k)=IMscore_id10;
    %names=[names,name(IMscore_id10)'];
    %name(IMscore_id10)'
end
L2L1SVM_median_IMrank=median(L2L1SVM_IMrank,2);%mean(L2L1SVM_IMrank,2);%average_IMrank(:,uniqueid_top10)   %;median(L2L1SVM_IMrank')';
[L2L1SVM_B_median_IMrank,L2L1SVM_I_median_IMrank] = sort(L2L1SVM_median_IMrank,'ascend');
%L2L1SVM_summary= [name(L2L1SVM_I_median_IMrank(1:15))'   num2cell(median(L2L1SVM_IMscore(L2L1SVM_I_median_IMrank(1:15),:),2))]
L2L1SVM_summary= [name(L2L1SVM_I_median_IMrank)'   num2cell(median(L2L1SVM_IMscore(L2L1SVM_I_median_IMrank,:),2))];
%%
L1L2SVM_IMscore=zeros(size(data,2),10);id_top10=zeros(10,10);L1L2SVM_IMrank=zeros(size(data,2),10);
for k=1:10
    L1L2SVM_IMscore(setdiff(1:size(data,2),removed_dim10{k}),k)=L1L2_model_liblinear{k, 1}.w/abs(L1L2_model_liblinear{k, 1}.bias);
    [B,I] = sort(abs(L1L2SVM_IMscore(:,k)),'descend');
    L1L2SVM_IMrank(I,k)=1:size(data,2);
    IMscore_id10=I(1:10);
    id_top10(:,k)=IMscore_id10;
end
L1L2SVM_median_IMrank=median(L1L2SVM_IMrank,2);
[L1L2SVM_B_median_IMrank,L1L2SVM_I_median_IMrank] = sort(L1L2SVM_median_IMrank,'ascend');
% L1L2SVM_summary= [name(L1L2SVM_I_median_IMrank(1:15))'   num2cell(median(L1L2SVM_IMscore(L1L2SVM_I_median_IMrank(1:15),:),2))]
L1L2SVM_summary= [name(L1L2SVM_I_median_IMrank)'   num2cell(median(L1L2SVM_IMscore(L1L2SVM_I_median_IMrank,:),2))];
%%
save('summary_2SVMLR_wynorm.mat','L2LR_summary','L1LR_summary','L2L1SVM_summary','L1L2SVM_summary')
