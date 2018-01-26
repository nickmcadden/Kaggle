import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

mean_elo = 1500
elo_width = 400
k_factor = 20

data_dir = '../input/'
df_reg = pd.read_csv(data_dir + 'RegularSeasonCompactResults.csv')

df_concat = df_reg
df_concat.drop(labels=[ 'Wscore', 'Lscore', 'Wloc', 'Numot'], inplace=True, axis=1)
df_concat.sort_values(by=['Season', 'Daynum'], inplace=True)

'''
le = LabelEncoder()
df_concat.Wteam = le.fit_transform(df_concat.Wteam)
df_concat.Lteam = le.fit_transform(df_concat.Lteam)
'''

df_concat.Wteam -= 1101
df_concat.Lteam -= 1101

def update_elo(winner_elo, loser_elo):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expected_win = expected_result(winner_elo, loser_elo)
    change_in_elo = k_factor * (1-expected_win)
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expect_a = 1.0/(1+10**((elo_b - elo_a)/elo_width))
    return expect_a

def update_end_of_season(elos):
    """Regression towards the mean

    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    diff_from_mean = elos - mean_elo
    elos -= diff_from_mean/2
    return elos

df_concat['w_elo_before_game'] = 0
df_concat['w_elo_after_game'] = 0
df_concat['l_elo_before_game'] = 0
df_concat['l_elo_after_game'] = 0
elo_per_season = {}
n_teams = 364
current_elos = np.ones(shape=(n_teams)) * mean_elo

df_concat['total_days'] = (df_concat.Season-1970)*365.25 + df_concat.Daynum

df_team_elos = pd.DataFrame(index=df_concat.total_days.unique(), 
                            columns=range(n_teams))
df_team_elos.iloc[0, :] = current_elos

current_season = df_concat.at[0, 'Season']

for row in df_concat.itertuples():
    if row.Season != current_season:
        # Write the beginning of new season ratings to a dict for later lookups.
        elo_per_season[current_season] = current_elos.copy()
        # Check if we are starting a new season.
        # Regress all ratings towards the mean
        current_elos = update_end_of_season(current_elos)
        current_season = row.Season
    idx = row.Index
    w_id = row.Wteam
    l_id = row.Lteam
    # Get current elos
    w_elo_before = current_elos[w_id]
    l_elo_before = current_elos[l_id]
    # Update on game results
    w_elo_after, l_elo_after = update_elo(w_elo_before, l_elo_before)

    # Save updated elos
    df_concat.at[idx, 'w_elo_before_game'] = w_elo_before
    df_concat.at[idx, 'l_elo_before_game'] = l_elo_before
    df_concat.at[idx, 'w_elo_after_game'] = w_elo_after
    df_concat.at[idx, 'l_elo_after_game'] = l_elo_after
    current_elos[w_id] = w_elo_after
    current_elos[l_id] = l_elo_after

    # Save elos to team DataFrame
    today = row.total_days
    df_team_elos.at[today, w_id] = w_elo_after
    df_team_elos.at[today, l_id] = l_elo_after

elo_per_season[current_season] = current_elos.copy()
df_team_strength = pd.DataFrame()

for season in range(2002, 2017):
    df_teams = pd.read_csv(data_dir + 'Teams.csv')
    df_team_strength = pd.concat([df_team_strength, pd.DataFrame({'season':season, 'team_id': range(1101,1465), 'strength': elo_per_season[season]})])
    team_strengths = pd.merge(df_teams, df_team_strength, left_on=["Team_Id"], right_on=["team_id"])
    team_strengths = team_strengths.sort_values(by=['season', 'strength'], ascending=[1,0])
    top20 = team_strengths[team_strengths["season"]==season][:20]
    print(top20)
team_strengths = team_strengths[["season", "team_id", "strength"]]
team_strengths.to_csv("../input/elo_strength_ratings" + '.csv', index=False)
