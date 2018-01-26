import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skellam

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

mean_elo = 65
k_factor = 1.2
non_conference_multiplier = 2

data_dir = '../input/'
df_reg = pd.read_csv(data_dir + 'RegularSeasonCompactResults.csv')

df_concat = df_reg
df_concat.sort_values(by=['Season', 'Daynum'], inplace=True)

df_concat.Wteam -= 1101
df_concat.Lteam -= 1101

def update_elo(winner_elo, loser_elo, winning_margin, non_conference_game):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    expected_win = expected_result(winner_elo, loser_elo, winning_margin)
    change_in_elo = k_factor * expected_win
    if non_conference_game:
        change_in_elo *= non_conference_multiplier
    winner_elo += change_in_elo
    loser_elo -= change_in_elo
    return winner_elo, loser_elo

def expected_result(elo_a, elo_b, winning_margin):
    """
    https://en.wikipedia.org/wiki/Elo_rating_system#Mathematical_details
    """
    px = skellam.cdf(winning_margin, elo_a, elo_b)
    pwm = skellam.pmf(winning_margin, elo_a, elo_b)
    expect_a = (px+pwm*0.5) - 0.3
    return expect_a

def update_end_of_season(elos):
    """Regression towards the mean

    Following 538 nfl methods
    https://fivethirtyeight.com/datalab/nfl-elo-ratings-are-back/
    """
    diff_from_mean = elos - mean_elo
    elos -= diff_from_mean/1.5
    return elos

df_concat['w_elo_before_game'] = 0
df_concat['w_elo_after_game'] = 0
df_concat['l_elo_before_game'] = 0
df_concat['l_elo_after_game'] = 0
elo_per_season = {}
n_teams = 364
current_elos = np.ones(shape=(n_teams)) * mean_elo

df_concat['total_days'] = (df_concat.Season-1970) * 365.25 + df_concat.Daynum

df_team_elos = pd.DataFrame(index=df_concat.total_days.unique())

for t in range(n_teams):
	df_team_elos.at[5498.75, t] = mean_elo

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
    margin = row.Wscore - row.Lscore
    non_conference_game = (row.Wloc=='N')
    # Get current elos
    w_elo_before = current_elos[w_id]
    l_elo_before = current_elos[l_id]
    # Update on game results
    w_elo_after, l_elo_after = update_elo(w_elo_before, l_elo_before, margin, non_conference_game)

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

# print out top 20 ratings for all seasons
for season in range(2002, 2018):
    df_teams = pd.read_csv(data_dir + 'Teams.csv')
    df_team_strength = pd.concat([df_team_strength, pd.DataFrame({'season':season, 'team_id': range(1101,1465), 'strength': elo_per_season[season]})])
    team_strengths = pd.merge(df_teams, df_team_strength, left_on=["Team_Id"], right_on=["team_id"])
    team_strengths = team_strengths.sort_values(by=['season', 'strength'], ascending=[1,0])
    top20 = team_strengths[team_strengths["season"]==season][:20]
    print(top20)

# write ratings to csv
team_strengths = team_strengths[["season", "team_id", "strength"]]
team_strengths.to_csv("../input/elo_strength_ratings" + '.csv', index=False)

df_team_elos = df_team_elos.interpolate()
df_team_elos = np.round(df_team_elos, 2)

df_season_daynum = df_concat[["Season", "Daynum"]].groupby(["Season","Daynum"]).size().reset_index()
df_team_elos["season"] = np.array(df_season_daynum["Season"])
df_team_elos["daynum"] = np.array(df_season_daynum["Daynum"])

df_team_elos.to_csv("../input/elo_daily_ratings" + '.csv', index=False)
