import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data_dir = '../input/'
df_massey = pd.read_csv(data_dir + 'massey_ordinals_2003-2016.csv')
df_massey = df_massey[df_massey["rating_day_num"]==121]

print(df_massey)
df_massey_avg_rank = df_massey.groupby(['season', 'team']).agg({'orank': 'mean'}).reset_index()

print(df_massey_avg_rank[:10])

df_teams = pd.read_csv(data_dir + 'Teams.csv')
for season in range(2002, 2018):
    team_strengths = pd.merge(df_teams, df_massey_avg_rank, left_on=["Team_Id"], right_on=["team"])
    team_strengths = team_strengths.sort_values(by=['season', 'orank'], ascending=[1,1])
    top20 = team_strengths[team_strengths["season"]==season][:20]
    print(top20)

team_strengths = team_strengths[["season", "team", "orank"]]
team_strengths.columns = ["season", 'team_id', "strength"]
team_strengths.to_csv("../input/massey_ratings" + '.csv', index=False)
