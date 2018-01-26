import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from string import replace

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data_dir = '../input/'
df_reg = pd.read_csv(data_dir + 'RegularSeasonDetailedResults.csv')
df_teams = pd.read_csv(data_dir + 'Teams.csv')

df_reg["non_conf"] = (df_reg["Wloc"] == 'N').astype(int)

df_reg1 = df_reg.copy()
df_reg2 = df_reg.copy()

df_reg1.columns = [replace(col, 'W', 'team1_') for col in df_reg1.columns]
df_reg1.columns = [replace(col, 'L', 'team2_') for col in df_reg1.columns]
df_reg1.columns = [replace(col, '_team', '') for col in df_reg1.columns]
df_reg2.columns = [replace(col, 'W', 'team2_') for col in df_reg2.columns]
df_reg2.columns = [replace(col, 'L', 'team1_') for col in df_reg2.columns]
df_reg2.columns = [replace(col, '_team', '') for col in df_reg2.columns]

df_reg = pd.concat([df_reg1, df_reg2], axis=0)

df_reg['team1_margin'] = df_reg['team1_score'] - df_reg['team2_score']
df_reg['team1_tempo'] = df_reg['team1_fga'] + df_reg['team1_fga3']
df_reg['team2_tempo'] = df_reg['team2_fga'] + df_reg['team2_fga3']
df_reg['team1_eff_off'] = df_reg['team1_score'] / df_reg['team1_tempo']
df_reg['team1_eff_def'] = df_reg['team2_score'] / df_reg['team2_tempo']

# re-order the columns
cols = list(df_reg.columns)
#cols.remove('team2')
cols.remove('team1_loc')
cols.remove('team2_loc')
cols.remove('Season')
cols.remove('team1')
cols.insert(1,'team1')
cols.insert(0,'Season')
df_reg = df_reg[cols]

df_reg.fillna(0, inplace=True)

df_reg_nc = df_reg[df_reg["non_conf"] == 1]
df_reg_nc.columns = [replace(col, 'team1_', 'team1_nc_') for col in df_reg_nc.columns]
df_reg_nc.columns = [replace(col, 'team2_', 'team2_nc_') for col in df_reg_nc.columns]

print(len(df_reg))
print(len(df_reg_nc))

df_aggs = df_reg.groupby(["Season","team1"]).agg({'team1_margin': 'mean'}).reset_index()
df_aggs_nc = df_reg_nc.groupby(["Season","team1"]).agg({'team1_nc_margin': 'mean'}).reset_index()

df_aggs.columns = [replace(col, 'team1_margin', 'opp_avg_margin') for col in df_aggs.columns]
df_aggs_nc.columns = [replace(col, 'team1_nc_margin', 'opp_avg_nc_margin') for col in df_aggs_nc.columns]

df_reg = pd.merge(df_reg, df_aggs, left_on=['Season', 'team2'], right_on=['Season', 'team1'], suffixes=('', '_y'))
df_reg["team1_index"] = df_reg["team1_margin"] + df_reg["opp_avg_margin"]

df_reg_nc = pd.merge(df_reg_nc, df_aggs_nc, left_on=['Season', 'team2'], right_on=['Season', 'team1'], suffixes=('', '_y'))
df_reg_nc["team1_nc_index"] = df_reg_nc["team1_nc_margin"] + df_reg_nc["opp_avg_nc_margin"]

df_aggs = df_reg.groupby(["Season","team1"]).mean().reset_index()
df_aggs_nc = df_reg_nc.groupby(["Season","team1"]).mean().reset_index()

df_aggs.columns = [replace(col, 'team1', 'agg_team1') for col in df_aggs.columns]
df_aggs.columns = [replace(col, 'team2', 'agg_team2') for col in df_aggs.columns]
df_aggs_nc.columns = [replace(col, 'team1', 'agg_team1') for col in df_aggs_nc.columns]
df_aggs_nc.columns = [replace(col, 'team2', 'agg_team2') for col in df_aggs_nc.columns]

print(df_aggs_nc.columns)

df_aggs_nc = df_aggs_nc[["Season", "agg_team1", "agg_team1_nc_margin", "opp_avg_nc_margin"]]

print(len(df_aggs))
print(len(df_aggs_nc))

#df_aggs = pd.merge(df_aggs, df_aggs_nc, how="left", left_on=['Season', 'agg_team1'], right_on=['Season', 'agg_team1'], suffixes=('', '_y'))
#print(df_aggs[df_aggs["agg_team1"] == 1246][["agg_team1_margin", "agg_team1_nc_margin", "agg_team1_index", "agg_team1_nc_index"]])

#df_aggs["agg_team1_nc_margin"] = np.where(np.isnan(df_aggs["agg_team1_nc_margin"]), df_aggs["agg_team1_margin"], df_aggs["agg_team1_nc_margin"])
#df_aggs["opp_avg_nc_margin"] = np.where(np.isnan(df_aggs["opp_avg_nc_margin"]), df_aggs["opp_avg_margin"], df_aggs["opp_avg_nc_margin"])

df_aggs.to_csv("../input/agg_features2.csv", index=False)

'''
# This part is where the cumulative aggs are made. Not enabled for now.

team_ids = list(pd.unique(df_teams["Team_Id"]))
seasons = list(pd.unique(df_reg["Season"]))

for team in team_ids:
	print(team)
	df_team_reg = df_reg[df_reg["team1"] == team]
	for season in seasons:
		df_team_season_reg = df_team_reg[df_team_reg["Season"] == season]
		a = np.array([range(1,len(df_team_season_reg)+1)] * 29).T
		df_team_season_reg.ix[:,3:] = df_team_season_reg.ix[:,3:].cumsum() / a

print(df_cumulative)
'''
