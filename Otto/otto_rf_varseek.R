# clear environment workspace
rm(list=ls())
gc()
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
# load data
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")
# randomise training data
train <- train_raw[sample(nrow(train_raw)),-1]
train$target <- factor(substr(train[,94],7,8))

# find 30 most important features
fit1 <- randomForest(target ~ ., data=train, importance=F, keep.forest=F, ntree=30, mtry=25)
x<-importance(fit1)
z<-order(-x)[1:30]

# create new truncated training set
train_bst_f <- train[,z[order(z)]]
mat_bst_f <- as.matrix(train_bst_f)
# permute features 
mat_big <- t(apply(mat_bst_f, 1, combn, 2, function(x) x[1]/x[2] ))
mat_big[is.finite(mat_big)==FALSE] <- 0
colnames(mat_big)<-combn(colnames(train_bst_f), 2, function(x) paste0(x[1],"d",x[2]))
mat_big<-round(mat_big,2)
df_big<-cbind(train[,-94],as.data.frame(mat_big),target=train$target)

fit <- randomForest(target ~ ., data=df_big, importance=F, keep.forest=F, ntree=25, mtry=25)
y<-importance(fit)
z2<-order(-y)[1:150]
train_final <- df_big[,z2[order(z2)]]
train_final <- cbind(train_final,target=train$target)

# install eta package
require(randomForest)
# set a unique seed number so you get the same results everytime you run the below model,
set.seed(12)
st<-Sys.time()
for (i in 1:1) {
  cv_sample <- sample(floor(nrow(train)*0.8))
  cv_train <- train[cv_sample,] 
  cv_val <- train[-cv_sample,] 
  # create a random forest model using the target field as the response and all 93 features as inputs
  fit <- randomForest(target ~ ., data=cv_train, importance=F, keep.forest=T, ntree=250, mtry=31)
  # use the random forest model to create a prediction
  predrf <- predict(fit,cv_val[,-ncol(cv_val)],type="prob")
  predrf <- cbind(predrf,cv_val[,ncol(cv_val)])
  rowTotals <- apply(predrf,1,function(x) max(log(x[x[10]]),-4))
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
