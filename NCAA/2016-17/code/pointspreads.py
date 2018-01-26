import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import norm

data_dir = '../input/'
df_tourneys = pd.read_csv(data_dir + 'TourneyCompactResults.csv')
df_pro_lines = pd.read_csv(data_dir + 'ThePredictionTrackerPointspreads.csv')
df_pro_lines = df_pro_lines.query('daynum >= 136 and season>=2013')
df_pro_lines["pred"] = norm.cdf(df_pro_lines["line"] / 12.1)

df_sbr_lines = pd.read_csv(data_dir + 'SBRLines.csv')
df_sbr_lines = df_sbr_lines.query('daynum >= 136 and season>=2013')
df_sbr_tourney1 = pd.merge(df_tourneys, df_sbr_lines, left_on=["Season","Daynum","Wteam"], right_on=["season","daynum","hometeam"])
df_sbr_tourney1 = df_sbr_tourney1[["Season","Daynum","Wteam","Lteam","home_close1"]]

df_sbr_tourney2 = pd.merge(df_tourneys, df_sbr_lines, left_on=["Season","Daynum","Lteam"], right_on=["season","daynum","hometeam"])
df_sbr_tourney2 = df_sbr_tourney2[["Season","Daynum","Wteam","Lteam","home_close1"]]
df_sbr_tourney2["home_close1"] = -df_sbr_tourney2["home_close1"]

df_sbr_lines = pd.concat([df_sbr_tourney1, df_sbr_tourney2])
df_sbr_lines["pred"] = norm.cdf(df_sbr_lines["home_close1"] / 12.1)

scores = []
for season in[2014,2015]:
	tourney_season = df_sbr_lines[df_sbr_lines["Season"] ==season]
	print(len(tourney_season))
	logloss = -np.sum(np.log(tourney_season["pred"]))/ len(tourney_season)
	print(season, logloss)
	scores.append(logloss)

scores = []
for season in[2013,2014,2015]:
	tourney_season = df_pro_lines[df_pro_lines["season"] ==season]
	print(len(tourney_season))
	logloss = -np.sum(np.log(tourney_season["pred"]))/ len(tourney_season)
	print(season, logloss)
	scores.append(logloss)

scores.append(0.558)
print(np.mean(scores))
