# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
load("/Users/nmcadden/Desktop/Kaggle/Otto/.RData")
load(".RData")
# load data
train <- read.csv("train.csv")
test <- read.csv("test.csv")
# remove id column so it doesn't get picked up by the random forest classifier
train2 <- train[,-1]

library(glmnet)

# set a unique seed number so you get the same results everytime you run the below model,
set.seed(12)
# create a linear model using the target field as the response and all 93 features as inputs
fit<- glmnet(as.matrix(train2[,1:93]), as.vector(train2$target), family="multinomial")

# use the  model to create a prediction
pred <- predict(fit,test)
submit <- data.frame(id = test$id, pred)

# calculate log loss
groundTruthLabel <- as.numeric(substr(test$target,7,8))
preds <- cbind(pred,groundTruthLabel)
rowTotals <- apply(preds,1,function(x) log(x[x[10]]))
logLoss <- sum(rowTotals)/length(rowTotals)*-1

# write submission file
write.csv(submit, file = "firstsubmit.csv", row.names = FALSE)
