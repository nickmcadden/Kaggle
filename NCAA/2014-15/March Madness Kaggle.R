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

ncaa_games <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/game_reg_basic.csv"))
analysis_games <- subset(ncaa_games,season >= 2008)[,c(1:8,43:45)]
ncaa_teams <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/teams.csv"))
ncaa_fixtures <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/fixtures.csv"))
ncaa_tourney <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/tourney_detailed_results.csv"))
ncaa_team_conf <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/ncaa_team_conf.csv"))
ncaa_conferences <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/ncaa_conferences.csv"))
agg_features <- as.data.table(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/agg_features.csv"))
point_spreads <- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/pointspreads.csv"))
leaderboard<- as.data.frame(read.csv("/Users/nmcadden/Desktop/Kaggle/NBA/leaderboard.csv"))

#make team ids sequential
ncaa_teams$team_id<- ncaa_teams$team_id - 1100
analysis_games$wteam <- analysis_games$wteam - 1100
analysis_games$lteam <- analysis_games$lteam - 1100
ncaa_tourney <- subset(ncaa_tourney,season >=2004)[,1:6]

home_wins<-subset(analysis_games, wloc=="H" & season>=2008)
away_wins<-subset(analysis_games, wloc=="A" & season>=2008)
xmean <- (sum(home_wins$wscore) + sum(away_wins$lscore)) / (nrow(home_wins)+nrow(away_wins))
ymean <- (sum(home_wins$lscore) + sum(away_wins$wscore)) / (nrow(home_wins)+nrow(away_wins))

analysis_games$margin <- analysis_games$wscore - analysis_games$lscore
a_games <-subset(analysis_games,lscore >=43 & margin <=38 & wscore <=115)

n<-nrow(ncaa_teams)
attparams=rep(5,n)
defparams=rep(1,n)
logLikAudit <- numeric()
attvec <- numeric()
defvec<-numeric()
seasons<-c(2008,2009,2010,2011,2012,2013,2014)

for (i in seasons) {
  game_window <- subset(a_games, season >= i & season <=i)
  game_window$yearlag <- i- game_window$season
  # Cap outlier scores
  scoredata <- game_window[,c("wscore","lscore","wteam","lteam","yearlag","xmean","ymean")]
  # Run the optimiser
  m1 <- do_admb("ncaa_exponential",
                data=c(list(nobs=nrow(scoredata),decaymod=1),scoredata),
                params=list(att=attparams,def=defparams),
                verbose=FALSE,
                run.opts=run.control(compile=TRUE,write_files=TRUE,checkparam="ignore",checkdata="write",clean="ignore"))
  # Store quality of fit
  #logLikperGame <- -logLik(m1)
  #logLikAudit <- c(logLikAudit,logLikperGame)
  # Add params to uber list
  attparams <- as.vector(coef(m1)[1:n])
  defparams <- as.vector(coef(m1)[(n+1):(n*2)]) 
  attvec<-c(attvec,attparams)
  defvec<-c(defvec,defparams)
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
  #odds <- bballpredict(fix,attparams,defparams)
  #oddsdf <- rbind(oddsdf,odds)
  # Calc loss function on tournament games
  fixtures <- subset(ncaa_tourney,season==i & daynum >= 136)
  fix <- fixtures[c("wteam","lteam")]-1100
  odds <- cbind(fixtures[,c("season","daynum","wteam","lteam")],as.data.frame(bballpredict(fix,attparams,defparams)))
  toddsdf <- rbind(toddsdf,odds)
}
s<-aggregate(-log(toddsdf[5]), by=list(toddsdf$season), FUN=mean, na.rm=TRUE)
cbind(mean(s$V1),sd(s$V1),mean(subset(s,Group.1>=2011)[,2]))

fixtures <- subset(ncaa_fixtures,season>=2011 & season<=2014)[1:3]
a <-cbind(fixtures,oddsdf[,1])
colnames(a) <- c("season","team1","team2","pred")
b<- data.frame(id=paste(a$season,a$team1,a$team2,sep="_"),pred=a$pred)
write.csv(b, file = "ncaa_submission2.csv",quote=FALSE,row.names=FALSE)
write.csv(as.vector(unique(ncaa_teams_ext$Conf)), file = "ncaa_conferences.csv",quote=FALSE,row.names=FALSE)

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
all_pred$bas3<-apply(all_pred,1,function(x) if(x[2]<=139) x[5] else if(x[2]>=152) ((x[5]*6)+0.5)/7 else ((x[5]*7)+0.5)/8)
all_pred<-as.data.table(all_pred)

tourn1<-subset(all_pred,daynum<=137)[,list(DixCol=-mean(log(V1)),LReg=-mean(log(glm)),RF=-mean(log(rf)),line=-mean(log(line)),line_avg=-mean(log(line_avg)),bas2=-mean(log(bas2)),bas3=-mean(log(bas3)),cnt=.N),by="season"]
tourn2<-subset(all_pred,daynum>137)[,list(DixCol=-mean(log(V1)),LReg=-mean(log(glm)),RF=-mean(log(rf)),line=-mean(log(line)),line_avg=-mean(log(line_avg)),bas2=-mean(log(bas2)),bas3=-mean(log(bas3)),cnt=.N),by="season"]
all_tourn<-all_pred[,list(DixCol=-mean(log(V1)),LReg=-mean(log(glm)),RF=-mean(log(rf)),sourced_avg=-mean(log(line_avg)),line=-mean(log(line)),bas=-mean(log(bas)),bas2=-mean(log(bas2)),bas3=-mean(log(bas3)),cnt=.N),by="season"]
rbind(colMeans(all_tourn),colMeans(subset(all_tourn,season>=2011)))
