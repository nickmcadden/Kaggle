# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
# load data
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")
# randomise training data
train <- train_raw[sample(nrow(train_raw)),-1]
train$target <- factor(substr(train[,94],7,8))

options( java.parameters = "-Xmx2g" )
# install eta package
library(extraTrees)
# set a unique seed number so you get the same results everytime you run the below model,
set.seed(12)
st<-Sys.time()
for (i in 1:2) {
  cv_sample <- sample(floor(nrow(train)*0.8))
  cv_train <- train[cv_sample,] 
  cv_val <- train[-cv_sample,] 
  fit <- extraTrees(as.matrix(cv_train[,-ncol(cv_train)]), cv_train[,ncol(cv_train)], ntree=300, mtry=29)
  predert <- predict(fit,as.matrix(cv_val[,-ncol(cv_val)]),probability=TRUE)
  fit <- randomForest(as.matrix(cv_train[,-ncol(cv_train)]), cv_train[,ncol(cv_train)], ntree=300, mtry=29)
  predert <- predict(fit,as.matrix(cv_val[,-ncol(cv_val)]),probability=TRUE)
  predert <- cbind(predert,cv_val[,ncol(cv_val)])
  rowTotals <- apply(predert,1,function(x) max(log(x[x[10]]),-4))
  logloss <- -sum(rowTotals)/length(rowTotals)
  print(paste("Multi Log Loss:",logloss))
  flush.console()
}
en<-Sys.time()
en-st

# use the random forest model to create a prediction
predrf <- predict(fit,test_raw[,-1])
colnames(predrf) <- unique(train[order(train$target),]$target)
pred2 <- as.factor(colnames(predrf)[max.col(predrf)])

confusionMatrix(pred2, train2$target)

# calculate log loss
groundTruthLabel <- as.numeric(substr(test$target,7,8))
predsrf <- cbind(predrf,groundTruthLabel)
rowTotals <- apply(predsrf,1,function(x) log(x[x[10]]))
logloss <- -sum(rowTotals)/length(rowTotals)
logloss

# write submission file
submit <- data.frame(id = test$id, predrf)
write.csv(submit, file = "submit_rf1.csv", row.names = FALSE,quote=FALSE)
