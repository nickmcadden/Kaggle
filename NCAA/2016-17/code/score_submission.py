import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import norm
# Input data files are available in the "../input/" directory.

df_sub = pd.read_csv('../output/' + 'lr3_stage1.csv', sep="[,_]").reset_index().ix[:,:4]
df_sub.columns = ["season", "team1", "team2", "pred"]

df_tourney = pd.read_csv('../input/' + 'TourneyCompactResults.csv')
df_tourney = df_tourney[df_tourney["Daynum"]>=136]

'''
df_lines = pd.read_csv('../input/' + 'ThePredictionTrackerPointspreads.csv')
df_lines = df_lines[df_lines["daynum"]>=136]
df_lines = df_lines[df_lines["daynum"]<=137]
df_lines["linepred"] = norm.cdf(df_lines["lineavg"] / 12.1)
'''

# df_tourney = pd.merge(df_tourney, df_lines, how = "left", left_on=["Season", "Daynum", "Wteam", "Lteam"], right_on =["season", "daynum", "wteam", "lteam"])

start_season = df_sub.ix[0, "season"]
end_season = df_sub.ix[len(df_sub)-1, "season"]

all_scores = []
for season in range(start_season, end_season+1):
	df_tourney_season = df_tourney[df_tourney["Season"]==season]
	ll_score = []
	for row in df_tourney_season.itertuples():
		sub_row = df_sub[df_sub["season"]==season]
		if row.Wteam < row.Lteam:
			sub_row = sub_row[sub_row["team1"]==row.Wteam]
			sub_row = sub_row[sub_row["team2"]==row.Lteam]
			pred = sub_row["pred"]
			# if not np.isnan(row.linepred):
				# pred = 0.5 * pred + 0.5 * row.linepred
			ll_score.append(-np.log(float(pred)))
		else:
			sub_row = sub_row[sub_row["team1"]==row.Lteam]
			sub_row = sub_row[sub_row["team2"]==row.Wteam]
			pred = sub_row["pred"]
			#if not np.isnan(row.linepred):
				#pred = 0.5 * pred + 0.5 * (1-row.linepred)
			ll_score.append(-np.log(float(1.0-pred)))
	#df_tourney_season_with_scores = pd.concat([df_tourney_season, pd.Series(ll_score)], axis=1)
	print(np.mean(ll_score))
	all_scores.append(np.mean(ll_score))
print(np.mean(all_scores))
