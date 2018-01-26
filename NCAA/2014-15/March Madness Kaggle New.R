# add rebounds per play rather than total rebounds
# function to standardise measures between 0 and 1
standardise <- function(dfcols) {
  dfavgs<-apply(dfcols,2,mean)
  dfsds<-apply(dfcols,2,sd)
  dfstd<-t(apply(dfcols,1,function(x){0.5+(((x-dfavgs)/dfsds)*0.19)}))
  return(dfstd)
}

skellam <- function(k,mu1,mu2){
  return(exp(-mu1-mu2)*((mu1/mu2)^(k/2))*besselI(2*sqrt(mu1*mu2),k))
}

# function to predict probability of home win, draw and away win for a game
bballpredict <- function(q, att, def) {
  n<-nrow(q)
  resultodds <- vector()
  neutralscore <- (xmean+ymean)/2
  for (j in 1:n) {
    lamda <- neutralscore * att[q[j,1]] * def[q[j,2]]
    mu <- neutralscore * att[q[j,2]]* def[q[j,1]]
    p_hw <- p_drw <- p_aw <-0
    # calculate probability matrix
    for (x in -75:0) {
      px <- skellam(x,lamda,mu)
      if(x<0) p_aw <- p_aw + px
      else p_aw <- p_aw + (px*p_aw)
    }
    resultodds <- c(resultodds,1-p_aw,p_aw)
  }
  return(matrix(resultodds,n,2,byrow=TRUE))
}

### To Do ###
library(R2admb)
library(RMySQL)
library(data.table)
library(boot)
library(caret)
library(randomForest)
#library(e1071)
#library(pROC)
setup_admb()

setwd('/Users/nmcadden/kaggle/NCAA/2014-15')
analysis_games <- as.data.frame(read.csv("data/game_reg_home_away.csv"))
ncaa_teams <- as.data.frame(read.csv("data/teams.csv"))
ncaa_fixtures <- as.data.frame(read.csv("data/fixtures.csv"))
ncaa_tourney <- as.data.frame(read.csv("data/tourney_detailed_results.csv"))
ncaa_team_conf <- as.data.frame(read.csv("data/ncaa_team_conf.csv"))
ncaa_conferences <- as.data.frame(read.csv("data/ncaa_conferences.csv"))
agg_features <- as.data.table(read.csv("data/agg_features.csv"))
point_spreads <- as.data.frame(read.csv("data/pointspreads.csv"))

#make team ids sequential
ncaa_teams$team_id<- ncaa_teams$team_id - 1100
analysis_games$hteam <- analysis_games$hteam - 1100
analysis_games$ateam <- analysis_games$ateam - 1100
analysis_games$distance <- as.numeric(analysis_games$distance)
ncaa_tourney <- subset(ncaa_tourney,season >=2004)[,1:6]
analysis_games$margin <- abs(analysis_games$hscore - analysis_games$ascore)

a_games <-subset(analysis_games,margin <=38 & hscore >=43 & ascore >=43 & hscore <=115 & ascore <=115)
a_games$hmean<- 62
a_games$amean<- 62
a_games$distance<- as.numeric(a_games$distance)/100
a_games$attendance<- as.numeric(a_games$attendance)/5000
hmean <- mean(a_games$hmean)
amean <- mean(a_games$amean)

n<-nrow(ncaa_teams)
attparams=rep(5,n)
defparams=rep(1,n)
logLikAudit <- numeric()
attvec <- numeric()
defvec<-numeric()
seasons<-c(2015)

# main routine
for (i in seasons) {
  game_window <- subset(a_games, season >= i & season <=i)
  game_window$yearlag <- i- game_window$season
  scoredata <- game_window[,c("hscore","ascore","hteam","ateam","yearlag","hmean","amean","distance","attendance","iconf","hbias")]
  # Run the optimiser
  m1 <- do_admb("admb_ncaa/ncaa_geo",
                data=c(list(nobs=nrow(scoredata), decaymod=1),scoredata),
                params=list(att=attparams,def=defparams, dcf=0.01, ccf=0.01, hcf=0.01, icf=0.01),
                verbose=TRUE,
                run.opts=run.control(compile=TRUE, write_files=TRUE, checkparam="ignore", checkdata="write", clean="ignore"))

  # Add params to uber list
  attparams <- as.vector(coef(m1)[1:n])
  defparams <- as.vector(coef(m1)[(n+1):(n*2)]) 
  attvec<-c(attvec,attparams)
  defvec<-c(defvec,defparams)
  dcoef<-coef(m1)["dcf"]
  dcoef<-coef(m1)["ccf"]
}

# collate all attack and defense params for all analysed weeks
attack_params <- matrix(attvec,n,length(seasons))
defense_params <- matrix(defvec,n,length(seasons))
rownames(attack_params) <- ncaa_teams$team_id
rownames(defense_params) <- ncaa_teams$team_id
colnames(attack_params) <- seasons
colnames(defense_params) <- seasons

toddsdf <- oddsdf <- data.frame(season=integer(),daynum=integer(),wteam=integer(),lteam=integer(),phw=numeric(),paw=numeric())
ap<-attack_params
dp<-defense_params
w<-numeric()
for (i in seasons) {
  cnt<- i-min(seasons)-1
  attparams <- as.vector(ap[,as.character(i)])
  defparams <- as.vector(dp[,as.character(i)])
  # Build all game combo odds
  fixtures <- subset(ncaa_fixtures,season==i)[1:3]
  fix <- fixtures[c("team1","team2")]-1100
  odds <- bballpredict(fix,attparams,defparams)
  oddsdf <- rbind(oddsdf,odds)
  # Calc loss function on tournament games
  fixtures <- subset(ncaa_tourney,season==i & daynum >= 136)
  if(nrow(fixtures)>0) {
    fix <- fixtures[c("wteam","lteam")]-1100
    odds <- cbind(fixtures[,c("season","daynum","wteam","lteam")],as.data.frame(bballpredict(fix,attparams,defparams)))
    toddsdf <- rbind(toddsdf,odds)
  }
}
s<-aggregate(-log(toddsdf[5]), by=list(toddsdf$season), FUN=mean, na.rm=TRUE)
cbind(mean(s$V1),sd(s$V1),mean(subset(s,Group.1>=2011)[,2]))

fixtures <- subset(ncaa_fixtures,season>=min(seasons) & season<=max(seasons))[1:3]
submission <-cbind(fixtures,oddsdf[,1])
submission <- subset(submission,season==2015)
colnames(submission) <- c("season","team1","team2","pred")
b<- data.frame(id=paste(submission$season,submission$team1,submission$team2,sep="_"),pred=a$pred)
write.csv(b, file = "ncaa_submission_f3.csv",quote=FALSE,row.names=FALSE)

# collate all attack and defense params for all analysed weeks
ncaa_teams$attparams <- attparams
ncaa_teams$defparams <- defparams


agg_features <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/agg_features.csv"))
agg_features_std <- cbind(agg_features[,1:3],standardise(agg_features[,4:length(agg_features)]))
ncaa_tourney <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/tourney_detailed_results.csv"))
ncaa_tourney <- subset(ncaa_tourney,season >=2003)[,1:6]
## split tournament result history into two sets to randomize the winners and losers (currently all winner presented first)
smp_size <- floor(0.5 * nrow(ncaa_tourney))
## set the seed to make your partition reproductible
set.seed(123)
random50pc_ind <- sample(seq_len(nrow(ncaa_tourney)), size = smp_size)
ncaa_tourney1 <- ncaa_tourney[random50pc_ind, ]
colnames(ncaa_tourney1)<- c("season","daynum","team1","team1scr","team2","team2scr")
ncaa_tourney1$team1win <- 1
ncaa_tourney1$wteam <- ncaa_tourney1$team1
ncaa_tourney2 <- ncaa_tourney[-random50pc_ind, ]
colnames(ncaa_tourney2)<- c("season","daynum","team2","team2scr","team1","team1scr")
ncaa_tourney2$team1win <- 0
ncaa_tourney2$wteam <- ncaa_tourney2$team2
ncaa_tourney_train <- rbind(ncaa_tourney1,ncaa_tourney2)

f<-ncaa_tourney_train
f<-merge(f,agg_features_std,by.x=c("season","team1"), by.y=c("season","team_id"))
f<-merge(f,agg_features_std,by.x=c("season","team2"), by.y=c("season","team_id"))
f<-f[order(f$season,f$daynum,f$wteam),]

# random split #
set.seed(907)
inTraining <- createDataPartition(f$team1win, p = .75, list = FALSE)
training <- f[ inTraining,][-c(1:6,8:9,32)]
testing  <- f[-inTraining,][-c(1:6,8:9,32)]
# time based split #
testing <- subset(f,season>=2008&daynum>=136)[-c(1:6,8:11,13,20,23:24,32:34,38,43,46:47)]
training <- subset(f,season<2008)[-c(1:6,8:11,13,20,23:24,32:34,38,43,46:47)]
# Model #
tc<-trainControl(method = "repeatedcv",number=5,repeats =3,classProbs=TRUE)
fit <- train(as.factor(team1win) ~ .,data=training,method = "glm",family=binomial)
p1<-predict(fit, newdata = testing)
logloss<- -sum(testing$team1win * log(p1) + abs(1-testing$team1win) * log(1-p1))/nrow(testing)
logloss

# random split #
set.seed(907)
inTraining <- createDataPartition(f$team1win, p = .75, list = FALSE)
training <- f[ inTraining,][-c(1:6,8:11,13,20,23:24,32:34,38,43,46:47)]
testing  <- f[-inTraining,][-c(1:6,8:11,13,20,23:24,32:34,38,43,46:47)]
# time based split #
testing <- subset(f,season>=2008&daynum>=136)[-c(1:6,8:11,13,20,23:24,32:34,38,43,46:47)]
training <- subset(f,season<2008)[-c(1:6,8:11,13,20,23:24,32:34,38,43,46:47)]
# Model #
tc<-trainControl(method = "repeatedcv",number=5,repeats =3,classProbs = TRUE, summaryFunction = twoClassSummary)
rf.fit <- train(as.factor(team1win) ~ .,data=training,ntree=1000, trControl = tc, metric="AUC")
p2<-predict(rf.fit, testing, type="prob")[,2] # predicted values
logloss<- -sum((testing$team1win) * log(p2) + abs(1-testing$team1win) * log(1-p2))/nrow(testing)
logloss

d<-cbind(toddsdf[,1:5],glm=(testing$team1win * p1 + abs(1-testing$team1win) * (1-p1)),rf=(testing$team1win * p2 + abs(1-testing$team1win) * (1-p2)))
pro_lines <- cbind(point_spreads[,1:9],data.frame(line=pnorm(point_spreads$line,0,11),line_avg=pnorm(point_spreads$lineavg,0,11)))
pro_lines <- subset(pro_lines,daynum>=136&season>=2008)

all_pred <-merge(d,pro_lines[,c(1:3,10:11)],by=c("season","daynum","wteam"),all.x=TRUE)
all_pred$line<-apply(all_pred,1,function(x) if(is.na(x[8])) x[5] else x[8])
all_pred$line_avg<-apply(all_pred,1,function(x) if(is.na(x[9])) x[5] else x[9])
all_pred$bas<-apply(all_pred,1,function(x) if(x[2]<=137) x[8] else x[5])
all_pred$bas2<-apply(all_pred,1,function(x) if(x[2]<=137) (x[5] + x[8]) / 2 else x[5]) 
all_pred$bas3<-apply(all_pred,1,function(x) if(x[5]>0.90&x[5]<0.96) 0.96 else if(x[5]<0.1&x[5]>0.04) 0.04 else if (x[2]<=137) x[5] else if (abs(x[5]-0.5)> 0) ((x[5]*8)+0.5)/9 else x[5])
all_pred<-as.data.table(all_pred)

tourn1<-subset(all_pred,daynum<=137)[,list(DixCol=-mean(log(V1)),LReg=-mean(log(glm)),RF=-mean(log(rf)),line=-mean(log(line)),line_avg=-mean(log(line_avg)),bas2=-mean(log(bas2)),bas3=-mean(log(bas3)),cnt=.N),by="season"]
tourn2<-subset(all_pred,daynum>137)[,list(DixCol=-mean(log(V1)),LReg=-mean(log(glm)),RF=-mean(log(rf)),line=-mean(log(line)),line_avg=-mean(log(line_avg)),bas2=-mean(log(bas2)),bas3=-mean(log(bas3)),cnt=.N),by="season"]
all_tourn<-all_pred[,list(DixCol=-mean(log(V1)),LReg=-mean(log(glm)),RF=-mean(log(rf)),sourced_avg=-mean(log(line_avg)),line=-mean(log(line)),bas=-mean(log(bas)),bas2=-mean(log(bas2)),bas3=-mean(log(bas3)),cnt=.N),by="season"]
rbind(colMeans(all_tourn),colMeans(subset(all_tourn,season>=2011)),apply(all_tourn,2,median))
