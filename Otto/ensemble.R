# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Desktop/Kaggle/Otto/lasagne/")
options(scipen=999)

nnetlist <- list.files(path = ".", pattern = "nnet.*", all.files = FALSE,
           full.names = FALSE, recursive = FALSE,
           ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)

ensmb <- as.matrix(read.csv(nnetlist[1])[,-1])
for (i in 1:20) {
  print(i)
  flush.console()
  en1 <- as.matrix(read.csv(nnetlist[i])[,-1])
  ensmb <- (ensmb * i + en1) / (i+1)
}
en100<-ensmb
en100<-en100/rowSums(en100)
en100<-cbind(id=1:nrow(en100),round(en100,5))
write.csv(en100, file = "nnet_ensemble6.csv", row.names = FALSE,quote=FALSE)


xgb2<-read.csv("../xgb2.csv")
nnet<-read.csv("nnet_ensemble6.csv")
ert<-read.csv("../ert_calibrated3.csv")
nnxgbert<-as.matrix(nnet*0.45+xgb2*0.45+ert*0.1)
nnxgbert<-nnxgbert[,-1]/rowSums(nnxgbert[,-1])
nnxgbert<-cbind(id=1:nrow(nnxgbert),round(nnxgbert,5))
write.csv(nnxgbert, file = "xgb+20nnet+ert.csv", row.names = FALSE,quote=FALSE)

xgb2<-read.csv("../xgb2.csv")
xgb3<-read.csv("../xgb_scikit_big.csv")
xgb<-as.matrix((xgb2+xgb3)/2)
xgb<-xgb[,-1]/rowSums(xgb[,-1])
xgb<-cbind(id=1:nrow(xgb),round(xgb,5))
write.csv(xgb, file = "2xgboosts.csv", row.names = FALSE,quote=FALSE)

nnet<-read.csv("nnet_ensemble5.csv")
xgb<-read.csv("2xgboosts.csv")
nnxgb<-as.matrix((nnet+xgb)/2)
nnxgb<-nnxgb[,-1]/rowSums(nnxgb[,-1])
nnxgb<-cbind(id=1:nrow(nnxgb),round(nnxgb,5))
write.csv(nnxgb, file = "2xgb+49nnet.csv", row.names = FALSE,quote=FALSE)




nnet<-read.csv("nnet_ensemble4.csv")
xgb<-read.csv("xgb2.csv")
svc<-read.csv("svc_calibrated.csv")
nnxgb<-as.matrix((nnet+xgb)/2)
nnxgb<-nnxgb[,-1]/rowSums(nnxgb[,-1])
nnxgb<-cbind(id=1:nrow(nnxgb),round(nnxgb,5))
write.csv(nnxgb, file = "xgb+nnet_2.csv", row.names = FALSE,quote=FALSE)

nnxgbsvc<-as.matrix(0.4*nnet+0.4*xgb+0.2*svc)
nnxgbsvc<-nnxgbsvc[,-1]/rowSums(nnxgbsvc[,-1])
nnxgbsvc<-cbind(id=1:nrow(nnxgbsvc),round(nnxgbsvc,5))
write.csv(nnxgbsvc, file = "xgb+nnet+svc.csv", row.names = FALSE,quote=FALSE)

nn_xgb<-as.matrix(nnet-xgb)
nn_xgb<-cbind(id=1:nrow(nn_xgb),round(nn_xgb,5))
write.csv(nn_xgb, file = "xgb+nnet_diff.csv", row.names = FALSE,quote=FALSE)

nn_svc<-as.matrix(nnet-svc)
nn_svc<-cbind(id=1:nrow(svc),round(nn_svc,5))
write.csv(nn_svc, file = "nn-svc-diff.csv", row.names = FALSE,quote=FALSE)

xgb_svc<-as.matrix(xgb-svc)
xgb_svc<-cbind(id=1:nrow(xgb_svc),round(xgb_svc,5))
write.csv(xgb_svc, file = "xgb-svc-diff.csv", row.names = FALSE,quote=FALSE)

require(data.table)
testdata<-data.frame(group=sample(letters,10e5,T),runif(10e5))

dti<-data.table(testdata)
#using native data.table
pti<-dti[dti[,list(val=sample(.I,10)),by="group"]$val]
zti<-dti[,list(val=sample(.I,10)),by="group"]
head(dti)
