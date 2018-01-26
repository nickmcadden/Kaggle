# clear environment workspace
rm(list=ls())
gc()
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
# load data
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")
# randomise training data
train <- train_raw[sample(nrow(train_raw)),]
train_ids <- train[,1]
train_class <- as.numeric(substr(train[,95],7,8))-1
train_features <- as.matrix(train[,-c(1,95)])
train_features <- matrix(as.numeric(train_features),nrow(train_features),ncol(train_features))
test <- as.matrix(test_raw[,-1])
test <- matrix(as.numeric(test),nrow(test),ncol(test))
require(xgboost)
require(methods)

# set a unique seed number so you get the same results everytime you run the below model,
set.seed(12)
#-------------Basic Training using XGBoost-----------------

print("training xgboost with sparseMatrix")
# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8,
              "eta" = 0.1,
              "max_depth" = 8,
              "min_child_weight" = 4,
              "subsample" = 0.875,
              "colsample_bytree" = 0.875)

# Run cross valication
#cv.nround = 50
#bst.cv = xgb.cv(param=param, data = train_features, label = train_class, nfold = 3, nrounds=cv.nround)

# Train the model
nround = 50
bst = xgboost(param=param, data = train_features, label = train_class, nrounds=nround)
xgb.importance(colnames(train_raw[,-c(1,95)]), model = bst)

# Make prediction
pred = predict(bst,test)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=3,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submit8.csv', quote=FALSE,row.names=FALSE)

confusionMatrix(pred2, t1$target)
# calculate log loss
groundTruthLabel <- as.numeric(substr(t1$target,7,8))
preds <- cbind(pred,groundTruthLabel)
rowTotals <- apply(preds,1,function(x) log(x[x[10]]))
logloss <- -sum(rowTotals)/length(rowTotals)
logloss

