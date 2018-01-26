# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
# load data
intrain <- read.csv("train.csv")
intest <- read.csv("test.csv")
# set a unique seed number so you get the same results everytime you run the below model
set.seed(12)
tsamples <- createDataPartition(intrain$target, p = .8, list = FALSE, groups = 9)
train <- intrain[tsamples,-1]
validate  <- intrain[-tsamples,-1]

# install randomForest package
library(adabag)
library(caret)

# create a random forest model using the target field as the response and all 93 features as inputs
fit <- boosting(target ~ ., data=train, boos=TRUE, mfinal=10)

# create a dotchart of variable/feature importance as measured by a Random Forest
importanceplot(fit)

# calculate log loss
pred_val <- predict.boosting(fit,validate)
groundTruthLabel <- as.numeric(substr(validate$target,7,8))
preds <- cbind(pred_val,groundTruthLabel)
rowTotals <- apply(preds,1,function(x) ifelse(x[x[10]]==0,-16,log(x[x[10]])))
logLoss <- sum(rowTotals)/length(rowTotals)*-1

# write submission file
pred_test <- predict(fit,test,type="prob")
submit <- data.frame(id = test$id, pred_test)
write.csv(submit, file = "firstsubmit.csv", row.names = FALSE)
