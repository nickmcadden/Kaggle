# install  package
require(randomForest)
require(unbalanced)
require(caret)

# function to calculate multinomal log loss
mLogLoss <- function(pred,trueclass) {
  rowTotals <- apply(cbind(pred,trueclass),1,function(x) max(log(x[x[10]]),-4))
  mlogloss <- -sum(rowTotals)/length(rowTotals)
  return(mlogloss)
}

# function to oversample the borderline cases of 2 classes
blSMOTE <- function(t,smClass,bgClass,oversmp,undersmp) {
  x <- subset(t, target==smClass | target==bgClass)
  # Make big class first cos this will be class 0
  x$target <- factor(x$target, levels = c(bgClass,smClass))
  levels(x$target)<-c("0","1")
  xfit <- randomForest(target ~ ., data=x, importance=F, keep.forest=T, ntree=30, mtry=21)
  predx <- cbind(data.frame(predict(xfit,type="prob")),x)
  predbad <- subset(predx, (predx[,1]<0.5 & target=="0") | (predx[,2]<0.5 & target=="1"))
  predgood <- subset(predx, (predx[,1]>=0.5 & target=="0") | (predx[,2]>=0.5 & target=="1"))
  x <- predbad[,-c(1,2)]
  bl <- ubSMOTE(x[,-ncol(x)], x$target, perc.over = oversmp, k = 5, perc.under = undersmp, verbose = TRUE)
  bl_smoted <- cbind(bl$X,target=bl$Y)
  train_smoted <-rbind(bl_smoted,predgood[,-c(1,2)])
  levels(train_smoted$target) <- c(bgClass,smClass)
  # merge with other original classes
  train_smoted <- rbind(train_smoted, subset(t, target!=smClass & target!=bgClass))
  train_smoted <- train_smoted[sample(nrow(train_smoted)),]
  train_smoted$target <- factor(as.character(train_smoted$target))
  return(train_smoted)
}

# clear environment workspace
rm(list=ls())
gc()
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
# load data
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")

# randomise training data
set.seed(12)
train <- train_raw[sample(nrow(train_raw)),-1]
train$target <- factor(substr(train[,94],7,8))

cv_sample <- createDataPartition(train$target,p=0.8,list = FALSE)
cv_sample <- cv_sample[sample(nrow(cv_sample))]
cv_train <- train[cv_sample,]
cv_val <- train[-cv_sample,]

# set a unique seed number so you get the same results everytime you run the below model,
st<-Sys.time()
set.seed(12)
# tsmote <- blSMOTE(cv_train,"3","2",30,100)
# tsmote <- cv_train
tsmote <- train
for (i in 1) {
  # Take percentage from each class
  strat1 <- ceiling(table(tsmote$target) * 0.91)
  # create a model
  c2c3fit <- randomForest(target ~ ., data=tsmote, importance=F, keep.forest=T, ntree=6000, mtry=32, replace=F, sampsize=strat1)
  predcv <- predict(c2c3fit,cv_val,type="prob")
  print(paste("Validation Multi Log Loss:",mLogLoss(predcv,cv_val$target)))
  predoob <- predict(c2c3fit,type="prob")
  print(paste("OOB Multi Log Loss:",mLogLoss(predoob,tsmote$target)))
  flush.console()
}
en<-Sys.time()
en-st

# write submission file
predrf <- predict(c2c3fit,test_raw[,-1],type="prob")
predrf <- format(predrf, digits=3,scientific=F)
colnames(predrf) <- paste0('Class_',1:9)
submit <- data.frame(id = test_raw$id, predrf)
write.csv(submit, file = "submit14.csv", row.names = FALSE,quote=FALSE)
