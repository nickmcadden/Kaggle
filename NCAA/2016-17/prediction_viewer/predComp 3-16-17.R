library(dplyr)
#library(readr)
library(ggplot2)
library(ggthemes)
library(stringr)
rm(list=ls())

setwd("/Users/nmcadden/kaggle/ncaa/2016-17/prediction_viewer/")

#grab out your predictions
Target<-read.csv("../output/ncaa_2017_stage2_mod1.csv")
names(Target) <- tolower(names(Target))

season <- 2017
teams <- read.csv("../input/Teams.csv")
tourneySeeds <- read.csv("../input/TourneySeeds.csv") %>% filter(Season == season) %>% select(-Season)
tourneySlots <- read.csv("../input/TourneySlots.csv") %>% filter(Season == season) %>% select(-Season)

# Get list of all files
setwd("C:/Users/pcalh_000/Desktop/March Madness/Spring 2017/Submissions")
# Load all files
allPredictions <- lapply(files, function(f) {dat <- read.csv(f); names(dat) <- tolower(names(dat)); dat})

# Don't count duplicated submissions multiple times (these might be popular scripts, benchmarks, "steal my entry")
allPredictions <- unique(allPredictions)


# Set seeds based on play-in results
W11_winner <- 1425
W16_winner <- 1291
Y16_winner <- 1413
Z11_winner <- 1243
tourneySeeds <- tourneySeeds %>%
  mutate(Seed = replace(Seed, Team == W11_winner, "W11")) %>%
  mutate(Seed = replace(Seed, Team == W16_winner, "W16")) %>%
  mutate(Seed = replace(Seed, Team == Y16_winner, "Y11")) %>%
  mutate(Seed = replace(Seed, Team == Z11_winner, "Z16"))

# Makes titles for faceted plots based on game id
idToTeams <- function(gameId, y) {
  teamIds <- str_extract_all(gameId,'[0-9]{4}')[[1]]
  team1 <- as.integer(teamIds[2])
  team2 <- as.integer(teamIds[3])
  team1Name <- substr(teams %>% filter(Team_Id == team1) %>% .$Team_Name, 1, 20)
  team2Name <- substr(teams %>% filter(Team_Id == team2) %>% .$Team_Name, 1, 20)
  return(as.character(paste0('<- ', team2Name,' vs. ',team1Name,' ->')))
}

# All submission probs
meltedSubmissions <- do.call("rbind", allPredictions)

#tag on yours with a flag
meltedSubmissions$Target<-0
Target$Target<-1
meltedSubmissions<-rbind(Target,meltedSubmissions)

# First round games
#firstRoundIds <- c(
#  "2017_1323_1343",
#  "2017_1423_1438",
#  "2017_1139_1457",
#  "2017_1211_1355",
#  "2017_1137_1452",
#  "2017_1190_1196",
#  "2017_1278_1292",
#  "2017_1321_1435",
#  "2017_1268_1462",
#  "2017_1291_1437",
#  "2017_1388_1433",
#  "2017_1345_1436",
#  "2017_1195_1199",
#  "2017_1439_1458",
#  "2017_1112_1315",
#  "2017_1235_1305"
#  )

firstRoundIds <- c(
  "2017_1276_1329",
  "2017_1124_1308",
  "2017_1116_1371",
  "2017_1211_1355",
  "2017_1233_1332",
  "2017_1240_1257",
  "2017_1374_1425",
  "2017_1314_1411",
  "2017_1166_1348",
  "2017_1242_1413",
  "2017_1173_1455",
  "2017_1181_1407",
  "2017_1153_1243",
  "2017_1274_1277",
  "2017_1246_1297",
  "2017_1266_1376"
  )


# Subset predictions
roundOf64 <- subset(meltedSubmissions, id %in% firstRoundIds)
roundOf64$id <- sapply(roundOf64$id, function(x) idToTeams(x, roundOf64))
roundOf64$id <- as.factor(roundOf64$id)

fancyPlot <- function(X, num_cols, filename) {
  p <- ggplot(data = X, aes(x = pred,fill="orange")) +
    facet_wrap(~id, ncol = 8) +
    geom_histogram(fill = "#20beff", binwidth=0.03) +
    #geom_vline(data=MyPreds,aes(xintercept = Median), color="#20beff") +
    geom_vline(data=MyPreds,aes(xintercept = pred), linetype = "longdash") +
    geom_vline(aes(xintercept = .5), linetype = "longdash") +
    ylab("Count") +
    xlab("Probability") +
    ggtitle('kaggle.com - March Machine Learning Mania 2017 (First Round)\n') +
    scale_x_continuous(breaks = c(0,0.25,0.5,0.75,1))  +
    theme(axis.ticks.y = element_blank()) +
    #theme_economist_white() +
    theme(strip.text.x = element_text(size = rel(1), color = "#47494d")) +
    theme(axis.title = element_text(size = rel(1.5), color = "#47494d")) + 
    theme(plot.title = element_text(size = rel(3), color = "#47494d", hjust = 0.5)) +
    theme(panel.background = element_rect(fill = '#fbfbfb')) + 
    theme(
      plot.background = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_blank()
    )
  ggsave(p, file = filename, width = 18, height = 11, dpi = 150)
}

#take your extra pred in the file out 
MyPreds <- subset(roundOf64, Target==1)
roundOf64<-subset(roundOf64, Target==0)
MyPreds$Median<-sapply(MyPreds$id,function(x) median(roundOf64$pred[roundOf64$id==x]))


fancyPlot(roundOf64, 2, 'kaggle_firstround_2017_day2.png')

