%RBF SVM with libsvm
clear; clc;
load('data4SVM_all_133_v7.mat','random10times','data','name','y')
data=sparse(data);
% needdelete=find(ismember(name,'"returnor"'))%[];
% data(:,needdelete)=[];%
% name(needdelete)=[];%
y(y==0)=-1;
TrainingRatio = 0.40%0.1%
N = 10;
N_fold = 3;

removed_dim10= cell(N,1);
RBFSVM_BESTC=zeros(N,1); RBFSVM_BESTG=zeros(N,1);
RBFSVM_AUC=zeros(N,1);
RBFSVM_model_libsvm= cell(N,1);
% Cs = 1% [ 10, 1, 0.1, 0.01, 0.001];%  [100,10, 1, 0.1, 0.01];%
% Sigmas =1%  [1000,100, 10, 1]; %[1000, 100, 10, 1]; % [10, 3, 1, 0.3];% Sigmas = [10]; is best for RBF(end)

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
    removed_dim=find(std(xtrain)< 1e-3);% 
    removed_dim10{k} =removed_dim;
    xtrain(:,removed_dim) = [];
    xtest(:,removed_dim) = [];
    t_mode = mode(xtrain);
    xtrain = xtrain - repmat(t_mode, size(xtrain,1),1);
    xtest = xtest - repmat(t_mode, size(xtest,1),1);
    %std_xtrain=range(xtrain);%
    std_xtrain=std(xtrain);%
    std_xtrain(std_xtrain==0)=1;
    xtrain=xtrain./repmat(std_xtrain, size(xtrain,1),1);%
    xtest=xtest./repmat(std_xtrain, size(xtest,1),1);
    
        %   RBFSVM SVM using libsvm  s for one run
        %-g gamma : set gamma in kernel function (default 1/num_features)
    %-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    % -v n: n-fold cross validation mode
    % -t kernel_type : set type of kernel function (default 2)
    % 	0 -- linear: u'*v
    % 	1 -- polynomial: (gamma*u'*v + coef0)^degree
    % 	2 -- radial basis function: exp(-gamma*|u-v|^2)
    % 	3 -- sigmoid: tanh(gamma*u'*v + coef0)
    % 	4 -- precomputed kernel (kernel values in training_set_file)
       % tic %
        RBFSVM_bestcv=0;
        for log10c = -1:2%0
            for log10g =-3:-1%-3
                cmd = ['-q -t 2 -v 3 -c ', num2str(10^log10c), ' -g ', num2str(10^log10g)];
                RBFSVM_cv=plotroc_libsvm(ytrain, xtrain, cmd);
                if (RBFSVM_cv >= RBFSVM_bestcv)
                    RBFSVM_bestcv = RBFSVM_cv; RBFSVM_BESTC(k) = 10^log10c;RBFSVM_BESTG(k) = 10^log10g;
                end
            end
        end
        fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log10c, log10g, RBFSVM_cv, RBFSVM_BESTC(k), RBFSVM_BESTG(k), RBFSVM_bestcv);

    %best C,G from last time run
%     RBFSVM_BESTC(k)=1;
%     RBFSVM_BESTG(k)=10^-3;
    RBFSVM_model_libsvm{k} = svmtrain_tw(ytrain,xtrain,['-q -t 2',' -c ', num2str(RBFSVM_BESTC(k)), ' -g ', num2str(RBFSVM_BESTG(k)),'-m 10000']);
    RBFSVM_AUC(k)=plotroc_libsvm(ytest,xtest, RBFSVM_model_libsvm{k});
    %which plotroc_libsvm
    RBFSVM_AUC(k)
end
toc;
time4 = cputime-time1;%

meanAUC_RBFSVM=  mean(RBFSVM_AUC)
stdAUC_RBFSVM= std(RBFSVM_AUC)

fname = sprintf('result_rbfSVM_20170911_%d_percent_wynorm_C1_G0001.mat',TrainingRatio*100);%_6feature
save(fname);









