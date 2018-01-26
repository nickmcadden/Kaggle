# clear environment workspace
rm(list=ls())
setwd("/Users/nmcadden/Kaggle/Caterpillar/")

### Load train and test
test  = read.csv("input/test_set.csv")
train = read.csv("input/train_set.csv")
tube  = read.csv("input/tube.csv")
specs = read.csv("input/specs.csv")

train$id = -(1:nrow(train))
test$cost = 0

train = rbind(train, test)

### combine component datasets
components = read.csv("input/components.csv")

comps <- components
for(f in dir("input/", pattern="^comp_")){
  print(dim(components))
  print(f)
  d = read.csv(paste0("input/", f))
  comps = merge(comps, d, by = "component_id", all.x=TRUE)
}

dupsdf <- comps[grep('\\.[xy]',names(comps))]
dups <- sub('\\.[xy].*','',names(dupsdf[order(names(dupsdf))]))
nondupsdf <- comps[-grep('\\.[xy]',names(comps))]

rm(newcomp)
for (col in unique(dups)) {
  dupcols <- comps[,grep(col,names(comps))]
  colvals <- apply(dupcols,1, function(x) ifelse(length(unique(x[!is.na(x)])), unique(x[!is.na(x)]), NA))
  if (exists("newcomp"))
    newcomp <- cbind(newcomp, colvals)
  else
    newcomp <- data.frame(colvals)
  colnames(newcomp)[ncol(newcomp)] <- col
}

nondupsdf<-nondupsdf[,-c(25,26,27,28,29,42,43)]
comps <- cbind(nondupsdf, newcomp)

bom = read.csv("input/bill_of_materials.csv")
ta_comps <- data.frame(tube_assembly_id=character(0), component_id=character(0), quantity=numeric(0))
for (i in 1:8) {
  y <- subset(bom[,c(1,i*2,i*2+1)],is.na(bom[,i*2])!=TRUE)
  colnames(y) <- c("tube_assembly_id", "component_id", "quantity")
  ta_comps <- rbind(ta_comps,y)
}

require(data.table)
ta_comps <- merge(comps,ta_comps,by="component_id")
ta_comps <- ta_comps[order(ta_comps$tube_assembly_id),]
ta_comps$cplx <- rowSums(!is.na(ta_comps))
ta_comps <- data.table(ta_comps)
ta_comps$unique_feature[is.na(ta_comps$unique_feature)]<- 'No'
ta_comps$unique_feature <- as.character(ta_comps$unique_feature)
ta_comps$unique_feature <- as.numeric(ta_comps$unique_feature=='Yes')
ta_comps$thread_size <- as.numeric(ta_comps$thread_size)
ta_aggs <- ta_comps[,list(ccnt=.N,cqty=sum(quantity),cunqtyp=length(unique(component_type_id)),cplx=sum(cplx),weight=sum(weight),sdwt=sd(weight),uf=sum(unique_feature), hs=sd(hex_size), sdl=sd(overall_length)),by=tube_assembly_id]

write.csv(ta_aggs, "input/ta_aggs.csv", row.names = FALSE, quote = FALSE)

train <- merge(train, ta_aggs, by = "tube_assembly_id", all.x = TRUE)
train <- merge(train, tube, by = "tube_assembly_id", all.x = TRUE)
train <- merge(train, specs, by = "tube_assembly_id", all.x = TRUE)
train <- merge(train, bom, by = "tube_assembly_id", all.x = TRUE)

### Clean NA values
for(i in 1:ncol(train)){
  if(is.numeric(train[,i])){
    train[is.na(train[,i]),i] = -1
  }else{
    train[,i] = as.character(train[,i])
    train[is.na(train[,i]),i] = "NAvalue"
    train[,i] = as.factor(train[,i])
  }
}

### Clean variables with too many categories
for(i in 1:ncol(train)){
  if(!is.numeric(train[,i])){
    freq = data.frame(table(train[,i]))
    freq = freq[order(freq$Freq, decreasing = TRUE),]
    train[,i] = as.character(match(train[,i], freq$Var1[1:30]))
    train[is.na(train[,i]),i] = "rareValue"
    train[,i] = as.factor(train[,i])
  }
}

test = train[which(train$id > 0),]
train = train[which(train$id < 0),]

### Randomforest
library(randomForest)

### Train randomForest on the whole training set
rf = randomForest(log(train$cost + 1)~., train[,-match(c("id", "cost"), names(train))], ntree = 20, do.trace = 2)

pred = exp(predict(rf, test)) - 1

submitDb = data.frame(id = test$id, cost = pred)
submitDb = aggregate(data.frame(cost = submitDb$cost), by = list(id = submitDb$id), mean)

write.csv(submitDb, "submit.csv", row.names = FALSE, quote = FALSE)

