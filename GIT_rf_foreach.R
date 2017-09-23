## 
# load all packages. you can use package foreach to speed up for large scale
library("foreach")
library("doSNOW")
library("randomForest")
library("ROCR")

## 
#load all numeric DATA
getwd()
df <- read.csv("data_2011to2014_2275452_107nocpt_median.csv", na.strings = c("")) 

## 
#Data preprocessing 1: delete highly related features
# drop ** which is highly related  Number of to **
drops <- unique(c("neurodef","cnscoma","pulembol","oprenafl","wndinfd","dehis","cnscva","oupneumo","othdvt","failwean","supinfec","othseshock","urninfec","othsysep","renainsf","orgspcssi","cdmi","othbleed","othgrafl","cdarrest","reintub"  ))#,"readmission" 
df<-df[ , !(names(df) %in% drops)]

#Data preprocessing 2:categorical features as.factor
Nunique<-rapply(df,function(x)length(unique(x)))
ignore<-c( "nneurodef","ncnscoma","npulembol","noprenafl","nwndinfd","ndehis","ncnscva","noupneumo","nothdvt","nfailwean","nsupinfec","nothseshock","nurninfec","nothsysep","nrenainsf","norgspcssi","ncdmi","nothbleed","nothgrafl","ncdarrest","nreintub" )
Nunique[ignore]<-0
# select columns
cols <-names(Nunique[Nunique>2 & Nunique<13])#
# c("race"      "transt"    "dischdest" "anesthes"  "surgspec"  "fnstatus2" "prsepis"   "wndclas"   "asaclas"   "admqtr"   )
df[,cols] <- data.frame(apply(df[cols], 2, as.factor))

# ###################################
# # #repeat 10 times for ramdom order 1:2275452;
#  random10times<- matrix(0, nrow = 10, ncol = nrow(df)) 
#  for (k in 1:10) {
#    random10times[k,] <- sample(1:nrow(df))
#  }
#  saveRDS(random10times, "random10times_all_2275452.rds")
#####################################
random10times <- readRDS("random10times_all_2275452.rds")


# multi-threads are used.
cl <- makeCluster(16, type="SOCK") # 16 – number of cores
registerDoSNOW(cl) # Register Backend Cores for Parallel Computing

readm_rate=sum(df$readmission)/nrow(df)
N_try<-nrow(df) #100000
xdata <- subset(df[1:N_try,], select=-c(readmission))
ydata <- subset(df[1:N_try,], select=c(readmission))


# training percentage. We randomly chose 40% of the patients in the dataset to form the training and validation set and keep the remaining patients as a test set.
TrainingRatio <-0.4
# repeat times. We repeat the training process 10 times according to a bootstrapping methodology and each time test on the test set, reporting the mean (Avg.) and standard deviation (Std.) of AUC
Nrep <-10

#time on
ptm <- proc.time() 

AUC <- matrix(0, nrow = Nrep, ncol = 1)
IMfactor <- matrix(0, nrow = dim(xdata)[2], ncol = Nrep)  
IMfactor2 <- matrix(0, nrow = dim(xdata)[2], ncol = Nrep)  

for (k in 1:Nrep){
  randPermIndex <-random10times[k,]
  index_selected_train <- randPermIndex[1:floor(dim(random10times)[2]*TrainingRatio)]
  index_selected_test <- randPermIndex[(1+floor(dim(random10times)[2]*TrainingRatio)):dim(random10times)[2]]
  
  xtrain <- xdata[index_selected_train,] 
  xtest <- xdata[index_selected_test,] 
  #   
  ytrain <- factor(ydata[index_selected_train,] )#ydata[index_selected_train] 
  ytest <- factor(ydata[index_selected_test,] ) #ydata[index_selected_test] 
  #####################
  # train balance    
  index_train_class2 <- which(ytrain == 1)#;%abnormal
  index_train_class1 <- which(ytrain == 0)  
  temp<-c(index_train_class2,index_train_class1[1:length(index_train_class2)])
  xtrain<-xtrain[temp,]
  ytrain <- ytrain[temp]#summary(ytrain)
  #####################
  # random forest with default parameters, you may tune if not good
  # system.time(
  surgery.rf <-foreach(ntree = rep(31, 16), .combine = combine, .packages = "randomForest") %dopar%{ #.multicombine = T, 
    randomForest(xtrain, ytrain, ntree = ntree, importance=TRUE)}
  #)
  # varImpPlot(  surgery.rf,type=2)  
  # varImpPlot(  surgery.rf,type=1)    
  # summary(surgery.preds <- predict(surgery.rf,  xtest, type = 'prob'))
  surgery.preds <- predict(surgery.rf,  xtest, type = 'prob')
  preds <- surgery.preds[,2]
  # Calculate the AUC value
  perf_AUC<- performance(prediction(preds, ytest ),"auc") 
  AUC[k]<- perf_AUC@y.values[[1]]
  IMfactor[,k]<- importance(surgery.rf ,type=1)
  IMfactor2[,k]<- importance(surgery.rf ,type=2)
}
# time count
proc.time() - ptm  

meanAUC=mean(AUC) #
print(meanAUC)
sdAUC=sd(AUC)# 
print(sdAUC)
#  close 关闭集群
stopCluster(cl)

name<-names(xdata)
# top topN=10 variable importance for Nrep=10 times
# method1
topN <- 10
top55 <- matrix(0, nrow = topN, ncol = Nrep)  
for (k in 1:Nrep){
  print(
    top55 [,k] <-  order(IMfactor[,k], decreasing=TRUE)[1:topN]
  )
}
top55unique <- (unique(c(top55)))
top55howmany <-rep(0, length(top55unique))
for (k in 1:length(top55unique)){
  print(
    top55howmany[k] <-length(which(top55==top55unique[k]))
  )
}
temp <- t(top55unique[which(top55howmany>=9)])

IMfactor_mean<-rowMeans(IMfactor,na.rm = TRUE)#IMfactor2
IMfactor_median<-apply(IMfactor, 1, median) 
#View((IMfactor_mean[top55unique]))
imp<-data.frame(top10howmany=top55howmany, IMfactormean=IMfactor_mean[top55unique],IMfactormedian=IMfactor_median[top55unique], row.names =name[top55unique])
imp<- imp[with(imp, order(-IMfactormedian, -top10howmany)),] 

##save variables
save(AUC,meanAUC,sdAUC,imp,surgery.rf,name,IMfactor, IMfactor2, file=paste("result_20170911", dim(xdata)[1], dim(xdata)[2],TrainingRatio*100,"percent_trainbalance_noCPT.RData", sep = "_"))

# print(surgery.rf)
