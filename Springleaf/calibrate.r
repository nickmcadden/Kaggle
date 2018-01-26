# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Kaggle/springleaf/output")
options(scipen=999)

model1 <- read.csv("xgb_ens___080216.csv")
model2 <- read.csv("xgb_2000_1866_0.01.csv")
model3 <- read.csv("nnet_0.4530_d50_h498_d28_h233_d4_e100_l0.001.csv")

hist(model1[,2],5,col="skyblue")
hist(model2[,2],5,col=scales::alpha('red',.5),add=T)

all <- cbind(rank(model1[,2]),rank(model2[,2]),rank(model3[,2]),(model1-model2)[,2])
y <- read.csv("../input/target.csv")
print(head(all,200))
print(mean(y[,1]))

en100<-ensmb
write.csv(en100, file = "xgb_ens7.csv", row.names = FALSE,quote=FALSE)
