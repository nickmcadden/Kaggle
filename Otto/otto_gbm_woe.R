# clear environment workspace
rm(list=ls())
gc()
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
load("/Users/nmcadden/Desktop/Kaggle/Otto/.RData")
# load data
train_raw <- read.csv("train.csv")
test_raw <- read.csv("test.csv")
# randomise training data
train <- train_raw[sample(nrow(train_raw)),]
train_ids <- train[,1]
train_class <- as.numeric(substr(train[,95],7,8)) 
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

gbmfit<-numeric()
predvec<-numeric()

for (i in 1:9) {
train_class <- as.numeric(substr(train[,95],7,8)) 
train_class[train_class!=i] <- 0
train_class[train_class==i] <- 1
t1 <- cbind(f1,target=train_class)

gbmfit<- gbm(target ~ ., 
              data=t1, 
              distribution="bernoulli",
              train.fraction = 0.8, 
              interaction.depth=8, 
              n.trees=500,
              shrinkage=0.05,
              bag.fraction = 0.7,
              n.minobsinnode = 8,
              keep.data=FALSE, 
              verbose=TRUE,
              n.cores=1)
  
  best_iter <- gbm.perf(gbmfit, method="test")
  pred <- predict(gbmfit, test, n.trees=best_iter, type="response")
  predvec <- c(predvec,pred)
}

predmat <- matrix(predvec,length(pred),9)
# write submission file
predsub <-  predmat/rowSums(as.data.frame(predmat))
colnames(predsub) <- unique(train[order(train$target),]$target)
submit <- data.frame(id = test$id, format(round(predsub, 4), nsmall = 4))
write.csv(submit, file = "submit5.csv", row.names = FALSE, quote=FALSE)


colnames(pred) <- unique(t1[order(t1$target),]$target)
pred2 <- as.factor(colnames(pred)[max.col(pred)])
confusionMatrix(pred2, t1$target)

# calculate log loss
groundTruthLabel <- as.numeric(substr(t1$target,7,8))
preds <- cbind(pred,groundTruthLabel)
rowTotals <- apply(preds,1,function(x) log(x[x[10]]))
logloss <- -sum(rowTotals)/length(rowTotals)
logloss

t23<- subset(train,substr(target,7,8)>=2&substr(target,7,8)<=3)
t23_class <- as.numeric(substr(t23[,95],7,8)) 
t23_class[t23_class==2] <- 1
t23_class[t23_class==3] <- 0
t23_feat<-t23[,-c(1,95)]
t23_feat[t23_feat==0] <- NA 
t23 <- cbind(t23_feat,target=t23_class)

# create a gbm model using the target field as the response and all 93 features as inputs
c2c3fit <- gbm(target ~ ., 
              data=t23,
              distribution="bernoulli",
              train.fraction = 0.8, 
              interaction.depth=9, 
              n.trees=500,
              shrinkage=0.1,
              bag.fraction = 0.7,
              n.minobsinnode = 10,
              keep.data=FALSE, 
              verbose=TRUE,
              n.cores=1)

best_iter <- gbm.perf(c2c3fit, method="test")

pred <- predict(c2c3fit, t23, n.trees=best_iter, type="response")
pred2 <- round(pred)
confusionMatrix(pred2, t23$target)

predtest <- predict(c2c3fit, test, n.trees=best_iter, type="response")
best_sub <- read.csv("submit4.csv")
newc2<-rowSums(best_sub[,3:4])*predtest
newc3<-rowSums(best_sub[,3:4])*(1-predtest)

best_sub$Class_2<-newc2
best_sub$Class_3<-newc3
submit <- data.frame(id = test$id, format(round(best_sub[,2:10], 4), nsmall = 4))
write.csv(submit, file = "submit6.csv", row.names = FALSE, quote=FALSE)


### one against one against rest ####
t23<- train
t23_class <- as.numeric(substr(t23$target,7,8))
t23_class[t23_class>3] <- 1
t23_class[t23_class<2] <- 1
t23_feat<-t23[,-c(1,95)]
t23_feat[t23_feat==0] <- NA 
t23 <- cbind(t23_feat,target=factor(t23_class))

# create a gbm model using the target field as the response and all 93 features as inputs
c2c3fit <- gbm(target ~ ., 
               data=t23,
               distribution="multinomial",
               train.fraction = 0.8, 
               interaction.depth=9, 
               n.trees=750,
               shrinkage=0.1,
               bag.fraction = 0.7,
               n.minobsinnode = 10,
               keep.data=FALSE, 
               verbose=TRUE,
               n.cores=1)

best_iter <- gbm.perf(c2c3fit, method="test")
pred <- as.data.frame(predict(c2c3fit, t23, n.trees=best_iter, type="response"))
colnames(pred) <- unique(t23[order(t23$target),]$target)
pred2 <- as.factor(colnames(pred)[max.col(pred)])
confusionMatrix(pred2, t23$target)

predtest <- as.data.frame(predict(c2c3fit, test, n.trees=best_iter, type="response"))
best_sub <- read.csv("submit4.csv")
newc2<-rowSums(best_sub[,3:4])/rowSums(predtest[,2:3])*predtest[,2]
newc3<-rowSums(best_sub[,3:4])/rowSums(predtest[,2:3])*predtest[,3]

best_sub$Class_2<-newc2
best_sub$Class_3<-newc3
submit <- data.frame(id = test$id, format(round(best_sub[,2:10], 4), nsmall = 4))
write.csv(submit, file = "submit7.csv", row.names = FALSE, quote=FALSE)
