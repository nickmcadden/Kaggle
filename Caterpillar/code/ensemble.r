# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Kaggle/Caterpillar/")

### Load train and test
q1  = read.csv("xgb0.232631178718.csv")
q2  = read.csv("xgb0.234734377069.csv")

q3 = q1+q2

qc2 = qc
qc2$lgqty = qc2$quantity^0.1

plot(qc2[2:3])