# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Kaggle/springleaf/output")
options(scipen=999)

xgblist <- list.files(path = ".", pattern = "xgb___080*", all.files = FALSE,
           full.names = FALSE, recursive = FALSE,
           ignore.case = FALSE, include.dirs = FALSE, no.. = FALSE)

print(xgblist)
ensmb <- read.csv(xgblist[1])
print(head(ensmb,20))
for (i in 2:2) {
  print(i)
  flush.console()
  en1 <- read.csv(xgblist[i])
  print(head(en1,20))
  ensmb <- sqrt(ensmb[,2] * en1[,2])
  print(head(ensmb,20))
}

en100<-ensmb
write.csv(en100, file = "xgb_ens5.csv", row.names = FALSE,quote=FALSE)
