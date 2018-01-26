
# clear environment workspace
rm(list=ls())
gc()
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/")
# load data
x <- read.csv("train.csv")
z <- read.csv("test.csv")

trainlg10 <- cbind(id=x[,1], round(log10(1+x[,-c(1,95)]),3), target=x[,95])
testlg10 <- cbind(id=z[,1], round(log10(1+z[,-1]),3))

write.csv(trainlg10, file = "trainlg10.csv", row.names = FALSE,quote=FALSE)
write.csv(testlg10, file = "testlg10.csv", row.names = FALSE,quote=FALSE)