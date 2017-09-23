##
library(xgboost)

## 
#load all numeric DATA
getwd()
df <- read.csv("data_2011to2014_2275452_107nocpt_median.csv", na.strings = c("")) 

## 
#Data preprocessing 1: delete highly related features
# drop ** which is highly related  Number of to **
drops <- unique(c("neurodef","cnscoma","pulembol","oprenafl","wndinfd","dehis","cnscva","oupneumo","othdvt","failwean","supinfec","othseshock","urninfec","othsysep","renainsf","orgspcssi","cdmi","othbleed","othgrafl","cdarrest","reintub"  ))#,"readmission" 
df<-df[ , !(names(df) %in% drops)]

##
N_try<-nrow(df) #100000
xdata <- subset(df[1:N_try,], select=-c(readmission))#df#subset(df[1:N_try,], select=-c(readmission))
ydata <- subset(df[1:N_try,], select=c(readmission))
# ###################################
# # #repeat 10 times for ramdom order 1:2275452;
#  random10times<- matrix(0, nrow = 10, ncol = nrow(df)) 
#  for (k in 1:10) {
#    random10times[k,] <- sample(1:nrow(df))
#  }
#  saveRDS(random10times, "random10times_all_2275452.rds")
#####################################
random5741 <- readRDS("random10times_all_2275452.rds")

# training percentage. We randomly chose 40% of the patients in the dataset to form the training and validation set and keep the remaining patients as a test set.
TrainingRatio <-0.4
# repeat times. We repeat the training process 10 times according to a bootstrapping methodology and each time test on the test set, reporting the mean (Avg.) and standard deviation (Std.) of AUC
Nrep <-10
# multi-threads are used.
Nthread<-16#28

AUC <- matrix(0, nrow = Nrep, ncol = 1)
#time on
ptm <- proc.time() 

for (k in 1:Nrep){#
  randPermIndex <-random5741[k,]
  index_selected_train <- randPermIndex[1:floor(dim(random5741)[2]*TrainingRatio)]
  index_selected_test <- randPermIndex[(1+floor(dim(random5741)[2]*TrainingRatio)):dim(random5741)[2]]
  
  xtrain <- xdata[index_selected_train,] 
  xtest <- xdata[index_selected_test,]    
  ytrain <- ydata[index_selected_train,] #factor(ydata[index_selected_train,] )#
  ytest <- ydata[index_selected_test,] #factor(ydata[index_selected_test,] ) #
  #####################
  #train balance   unique(ydata)  summary(ydata)
  index_train_class2 <- which(ytrain == 1)#;%abnormal
  index_train_class1 <- which(ytrain == 0)
  
  temp<-c(index_train_class2,index_train_class1[1:length(index_train_class2)])
  xtrain<-xtrain[temp,]
  ytrain <- ytrain[temp]#summary(ytrain)  
  #####################  
  dtrain <- xgb.DMatrix(data = as.matrix(xtrain), label=as.matrix(ytrain) )
  dtest <- xgb.DMatrix(data = as.matrix(xtest), label= as.matrix(ytest))  
  watchlist <- list(train=dtrain, test=dtest)  
  #   bst <- xgb.train(data=dtrain, max.depth=3, eta=0.2, nthread = Nthread, nround=500, watchlist=watchlist, eval.metric = "error", eval.metric = "auc", objective = "binary:logistic")
  ######################
  #CROSS-VALIDATION: RandomizedSearchCV
  best_param = list()
  best_seednumber = 6983#1234
  best_logloss = 0#Inf
  best_logloss_index = 0
  
  for (iter in 1:50) {
    param <- list( #silent = 0,
      eval_metric = "auc",
      max_depth = sample(7:12, 1),#c(2, 4, 6, 8, 10),#
      eta = runif(1, .01, .03),#runif(1, .01, .1)
      #gamma = runif(1, 0.0, 0.2), 
      subsample = runif(1, .6, 1),
      colsample_bytree = runif(1, .5, 1),#runif(1, .4, 1), 
      min_child_weight = sample(2:10, 1)#sample(1:10, 1)#,#min_child_weight = sample(1:40, 1),
      #max_delta_step = sample(1:10, 1) #Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
    )
    cv.nround = 1000
    cv.nfold = 3
    seed.number = sample.int(10000, 1)[[1]]
    set.seed(seed.number)
    mdcv <- xgb.cv(data=dtrain, params = param, nthread=Nthread, 
                   nfold=cv.nfold, nrounds=cv.nround,
                   verbose = 0, early_stopping_rounds=10, maximize=TRUE) # verbose = T, early.stop.round=8, maximize=FALSE)
    # print(mdcv$ evaluation_log$ test_auc_mean)
    # str(mdcv) 
    # mdcv[, max(test.auc.mean)]
    max_loss = max(mdcv$ evaluation_log$ test_auc_mean)
    max_loss_index = which.max(mdcv$ evaluation_log$ test_auc_mean)
    
    if (max_loss > best_logloss) {
      best_logloss = max_loss
      best_logloss_index = max_loss_index
      best_seednumber = seed.number
      best_param = param
      #       print(best_logloss)
      #       print(best_param)
    }
  }
  
  nround = best_logloss_index
  set.seed(best_seednumber) 
  bst <- xgb.train(data=dtrain, params=best_param, nrounds=nround, nthread = Nthread, watchlist=watchlist,  print_every_n = 10L, objective = "binary:logistic")  
  importance_matrix <- xgb.importance(feature_names = names(df),,model = bst)
  #str(importance_matrix)
  AUC[k]<-tail(bst$ evaluation_log$ test_auc, n=1)  
  #   xgb.plot.importance(importance_matrix = importance_matrix)   
  #   xgb.dump(bst, with.stats = T)
  save(list=c("bst","importance_matrix","nround","best_param","best_seednumber","AUC"), file = paste("xgboost", k,".Rda", sep = "_")) #load("parameters_weekday.Rda")  
  file = paste("xgboost", k,".Rda", sep = "_")
}
#time count
proc.time() - ptm  

meanAUC=mean(AUC) #
print(meanAUC)
sdAUC=sd(AUC)# 
print(sdAUC)

for (k in 1:10){
  load( paste("xgboost", k,".Rda", sep = "_"))
  print(best_param$max_depth)
  #print(best_param$min_child_weight)#nround
  #print(best_param$eta)
  #print(best_param$subsample)
  #print(best_param$colsample_bytree)
  #print(nround)
  #,"importance_matrix","nround","best_param","best_seednumber"
}

