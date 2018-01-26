# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Kaggle/springleaf/")

### Load train and test
train  = read.csv("train_head.csv")
str(train)