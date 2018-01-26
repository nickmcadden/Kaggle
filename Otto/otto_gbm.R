# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
# load data
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")
# randomise training data
train <- train_raw[sample(nrow(train_raw)),]
train_ids <- train[,1]
train_class <- train_class <- as.numeric(substr(train[,95],7,8))
train_features <- train[,-c(1,95)]
test<-test_raw
test[test==0] <-NA
f1 <- train_features
f1[f1==0] <- NA
t1 <- cbind(f1,target=train_class)

library(gbm)
library(caret)

# set a unique seed number so you get the same results everytime you run the below model,
set.seed(12)
# create a random forest model using the target field as the response and all 93 features as inputs
gbmfit <- gbm(target ~ ., 
              data=t1, 
              distribution="multinomial",
              train.fraction = 0.8, 
              interaction.depth=9, 
              n.trees=1000,
              shrinkage=0.05,
              bag.fraction = 0.7,
              n.minobsinnode = 10,
              keep.data=FALSE, 
              verbose=TRUE,
              n.cores=1)

# add trees
gbmfit <- gbm.more(gbmfit,n.new.trees=120)

best_iter <- gbm.perf(gbmfit, method="test")

pred <- as.data.frame(predict(gbmfit, t1, n.trees=300, type="response"))
colnames(pred) <- unique(t1[order(t1$target),]$target)

pred2 <- as.factor(colnames(pred)[max.col(pred)])
confusionMatrix(pred2, t1$target)

# calculate log loss
groundTruthLabel <- as.numeric(substr(t1$target,7,8))
preds <- cbind(pred,groundTruthLabel)
rowTotals <- apply(preds,1,function(x) log(x[x[10]]))
logloss <- -sum(rowTotals)/length(rowTotals)
logloss

# write submission file
predsub <- as.data.frame(predict(gbmfit, test, n.trees=750, type="response"))
colnames(predsub) <- unique(train[order(train$target),]$target)
submit <- data.frame(id = test$id, format(round(predsub, 4), nsmall = 4))
write.csv(submit, file = "submit4.csv", row.names = FALSE, quote=FALSE)

t23<- subset(train,substr(target,7,8)>=2&substr(target,7,8)<=3)
t23$target <- as.numeric(substr(t23$target,7,8))-2
# randomise training data
t23_ids <- t23[,1]
t23_class <- t23[,95]
t23_features <- t23[,-c(1,95)]
t23_features[t23_features==0] <- NA

t23 <- cbind(t23_features,target=t23_class)

# create a gbm model using the target field as the response and all 93 features as inputs
c2c3fit <- gbm(target ~ ., 
              data=t23,
              distribution="bernoulli",
              train.fraction = 0.7, 
              interaction.depth=8, 
              n.trees=500,
              shrinkage=0.1,
              bag.fraction = 0.8,
              n.minobsinnode = 8,
              keep.data=FALSE, 
              verbose=TRUE,
              n.cores=1)

best_iter <- gbm.perf(c2c3fit, method="test")

pred <- predict(c2c3fit, t23_features, n.trees=500, type="response")
pred2 <- round(pred)
confusionMatrix(pred2, t23$target)
#plot(c2c3fit,6,n.trees=300,type="response")

cust1 <- predict(c2c3fit, f1, n.trees=500, type="response")


f4<-cbind(f1,nfeat1=round(cust1))
t4 <- cbind(f4,target=train_class)
# set a unique seed number so you get the same results everytime you run the below model,
set.seed(12)
# create a random forest model using the target field as the response and all 93 features as inputs
gbmfit <- gbm(target ~ ., 
              data=t1, 
              distribution="multinomial",
              train.fraction = 0.8, 
              interaction.depth=9, 
              n.trees=750,
              shrinkage=0.05,
              bag.fraction = 0.7,
              n.minobsinnode = 10,
              keep.data=TRUE, 
              verbose=TRUE,
              n.cores=1)

plot(c2c3fit,6,n.trees=300,type="response")

