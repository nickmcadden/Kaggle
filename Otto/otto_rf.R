# install  package
require(randomForest)
require(unbalanced)
require(caret)

# function to standardise measures between 0 and 1
standardise2 <- function(dfcols) {
  dfmax<-apply(dfcols,2,function(x) max(x))
  dfstd<-t(apply(dfcols,1,function(x) x/dfmax))
  return(dfstd)
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
# Take 63.2% from each class
strat1 <- ceiling(table(train$target) * 0.77)
st<-Sys.time()
for (i in 1) {
  # create a random forest model using the target field as the response and all 93 features as inputs
  #fit <- randomForest(target ~ ., data=cv_train, importance=F, keep.forest=T, ntree=100, mtry=27)
  fit <- randomForest(target ~ ., data=train, importance=F, keep.forest=T, ntree=100, mtry=21, replace=F, sampsize=strat1)
  # use the random forest model to create a prediction
  predrf <- predict(fit,type="prob")
  predrf <- cbind(predrf[,1:9],train[,ncol(train)])
  rowTotals <- apply(predrf,1,function(x) max(log(x[x[10]]),-4))
  logloss <- -sum(rowTotals)/length(rowTotals)
  print(paste("Multi Log Loss:",logloss))
  flush.console()
}
en<-Sys.time()
en-st

t23<- subset(train,as.numeric(target)>=2&as.numeric(target)<=3)
# randomise training data
t23_class <- as.numeric(t23[,ncol(t23)])
t23_features <- t23[,-ncol(t23)]
t23_class[t23_class==2] <- 0
t23_class[t23_class==3] <- 1
t23 <- cbind(t23_features,target=as.factor(t23_class))
tt<-ubSMOTE(t23_features, factor(t23_class), perc.over = 100, k = 5, perc.under = 202, verbose = TRUE)
tt_class <- as.character(tt$Y)
tt_features <- tt$X
tt_class[tt_class=='0'] <- '2'
tt_class[tt_class=='1'] <- '3'
tt_all <- cbind(tt_features,target=as.factor(tt_class))
ty<- subset(train,as.numeric(target)!=2&as.numeric(target)!=3)
train2<-rbind(tt_all,ty)
train2 <- train2[sample(nrow(train2)),]
train2$target <- factor(as.character(train2$target))

# Take 63.2% from each class
strat1 <- ceiling(table(train2$target) * 0.77)
# set a unique seed number so you get the same results everytime you run the below model,
set.seed(12)
st<-Sys.time()
# create a model using the target field as the response and all 93 features as inputs
c2c3fit <- randomForest(target ~ ., data=train2, importance=F, keep.forest=T, ntree=100, mtry=21, replace=F, sampsize=strat1)

predrf <- predict(c2c3fit,type="prob")
predrf <- cbind(predrf[,1:9],train2[,ncol(train2)])
rowTotals <- apply(predrf,1,function(x) max(log(x[x[10]]),-4))
logloss <- -sum(rowTotals)/length(rowTotals)
print(paste("Multi Log Loss:",logloss))
flush.console()
en<-Sys.time()
en-st


# write submission file
predrf <- predict(c2c3fit,test_raw,type="prob")
colnames(predrf) <- paste0('Class_',1:9)
submit <- data.frame(id = test_raw$id, predrf)
write.csv(submit, file = "submit13.csv", row.names = FALSE,quote=FALSE)
