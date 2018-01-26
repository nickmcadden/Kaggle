import sys
import pandas as pd
import numpy as np
import os.path
import time
from sklearn.preprocessing import LabelEncoder

def get_rounds_vec(days, daytoround):
	vec = np.array([daytoround[day] for day in days])
	return vec

def load(m_params, stage_1):
	train = pd.read_csv("../input/TourneyDetailedResults.csv")
	train_aggs = pd.read_csv("../input/agg_features2.csv")
	team_attendance = pd.read_csv("../input/team_attendance.csv")
	conference_strengths = pd.read_csv("../input/conference_strengths.csv")
	elo_strength_ratings = pd.read_csv("../input/elo_strength_ratings.csv")
	tourney_geog = pd.read_csv("../input/TourneyGeog_Thru2016.csv")

	team_conference_strengths = pd.merge(team_conference_geog, conference_strengths, on="conf")[["team_id", "conf_strength"]]

	if stage_1:
		test = pd.read_csv("../input/sample_submission_stage1.csv", sep="[,_]").reset_index().ix[:,:3]
	else:
		test = pd.read_csv("../input/sample_submission_stage2.csv", sep="[,_]").reset_index().ix[:,:3]
	test.columns = ["season", "team1", "team2"]

	train = train.ix[:,:6]

	val = train[train["Season"] >= m_params['start_year']]
	val = val[val["Season"] <= m_params['end_year']]
	val = val[val["Daynum"] >= 136]

	train = train[train["Season"] < m_params['start_year']]

	## split tournament result history into two sets to randomize the winners and losers (currently all winner presented first)
	ix_50pc = np.random.rand(len(train)) < 0.5
	train_1 = train[ix_50pc]
	train_1.columns = ["season","daynum","team1","team1scr","team2","team2scr"]
	train_1["target"] = train_1["team1scr"] - train_1["team2scr"]
	train_2 = train[-ix_50pc]
	train_2.columns = ["season","daynum","team2","team2scr","team1","team1scr"]
	train_2["target"] = train_2["team1scr"] - train_2["team2scr"]
	train = pd.concat([train_1, train_2], axis=0)

	train = pd.merge(train, train_aggs, left_on=["season","team1"], right_on=["Season","agg_team1"])
	train = pd.merge(train, train_aggs, left_on=["season","team2"], right_on=["Season","agg_team1"])
	train = pd.merge(train, team_attendance, how="left", left_on="team1", right_on="team_id")
	train = pd.merge(train, team_attendance, how="left", left_on="team2", right_on="team_id")
	train = pd.merge(train, team_conference_strengths, left_on="team1", right_on="team_id")
	train = pd.merge(train, team_conference_strengths, left_on="team2", right_on="team_id")
	train = pd.merge(train, elo_strength_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	train = pd.merge(train, elo_strength_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	train["index_diff"] = train["agg_team1_index_x"] - train["agg_team1_index_y"]
	train["conf_strength_diff"] = train["conf_strength_x"] - train["conf_strength_y"]
	train["strength_diff"] = train["strength_x"] - train["strength_y"]

	val.columns = ["season","daynum","team1","team1scr","team2","team2scr"]
	val["target"] = val["team1scr"] - val["team2scr"]

	val = pd.merge(val, train_aggs, left_on=["season","team1"], right_on=["Season","agg_team1"])
	val = pd.merge(val, train_aggs, left_on=["season","team2"], right_on=["Season","agg_team1"])
	val = pd.merge(val, team_attendance, how="left", left_on="team1", right_on="team_id")
	val = pd.merge(val, team_attendance, how="left", left_on="team2", right_on="team_id")
	val = pd.merge(val, team_conference_strengths, left_on="team1", right_on="team_id")
	val = pd.merge(val, team_conference_strengths, left_on="team2", right_on="team_id")
	val = pd.merge(val, elo_strength_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	val = pd.merge(val, elo_strength_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	val["index_diff"] = val["agg_team1_index_x"] - val["agg_team1_index_y"]
	val["conf_strength_diff"] = val["conf_strength_x"] - val["conf_strength_y"]
	val["strength_diff"] = val["strength_x"] - val["strength_y"]

	val_inv = val.copy()
	val_inv = val_inv[["season","daynum","team1","team1scr","team2","team2scr"]]
	val_inv["team1"] = val["team2"]
	val_inv["team2"] = val["team1"]
	val_inv["team1scr"] = val["team2scr"]
	val_inv["team2scr"] = val["team1scr"]
	val_inv["target"] = val_inv["team1scr"] - val_inv["team2scr"]

	val_inv = pd.merge(val_inv, train_aggs, left_on=["season","team1"], right_on=["Season","agg_team1"])
	val_inv = pd.merge(val_inv, train_aggs, left_on=["season","team2"], right_on=["Season","agg_team1"])
	val_inv = pd.merge(val_inv, team_attendance, how="left", left_on="team1", right_on="team_id")
	val_inv = pd.merge(val_inv, team_attendance, how="left", left_on="team2", right_on="team_id")
	val_inv = pd.merge(val_inv, team_conference_strengths, left_on="team1", right_on="team_id")
	val_inv = pd.merge(val_inv, team_conference_strengths, left_on="team2", right_on="team_id")
	val_inv = pd.merge(val_inv, elo_strength_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	val_inv = pd.merge(val_inv, elo_strength_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	val_inv["index_diff"] = val_inv["agg_team1_index_x"] - val_inv["agg_team1_index_y"]
	val_inv["conf_strength_diff"] = val_inv["conf_strength_x"] - val_inv["conf_strength_y"]
	val_inv["strength_diff"] = val_inv["strength_x"] - val_inv["strength_y"]

	test = pd.merge(test, train_aggs, left_on=["season","team1"], right_on=["Season","agg_team1"])
	test = pd.merge(test, train_aggs, left_on=["season","team2"], right_on=["Season","agg_team1"])
	test = pd.merge(test, team_attendance, how="left", left_on="team1", right_on="team_id")
	test = pd.merge(test, team_attendance, how="left", left_on="team2", right_on="team_id")
	test = pd.merge(test, team_conference_strengths, left_on="team1", right_on="team_id")
	test = pd.merge(test, team_conference_strengths, left_on="team2", right_on="team_id")
	test = pd.merge(test, elo_strength_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	test = pd.merge(test, elo_strength_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	test["index_diff"] = test["agg_team1_index_x"] - test["agg_team1_index_y"]
	test["conf_strength_diff"] = test["conf_strength_x"] - test["conf_strength_y"]
	test["strength_diff"] = test["strength_x"] - test["strength_y"]

	test.sort(['season', 'team1', 'team2'], inplace=True)

	train_labels = train['target']

	val = val.sort_values(by=["season", "daynum", "team1", "team2"]).reset_index()
	val_inv = val_inv.sort_values(by=["season", "daynum", "team2", "team1"]).reset_index()
	val_labels = val['target']
	val_inv_labels = val_inv['target']

	train.drop(['daynum','team1', 'team2', 'team_id_x', 'team_id_y', 'team_name_x', 'team_name_y', 'team1scr', 'team2scr', 'target', 'Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)
	val.drop(['index', 'team1', 'team2', 'team_id_x', 'team_id_y', 'daynum', 'team_name_x', 'team_name_y', 'team1scr', 'team2scr', 'target', 'Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)
	val_inv.drop(['index', 'team1', 'team2', 'team_id_x', 'team_id_y', 'daynum', 'team_name_x', 'team_name_y', 'team1scr', 'team2scr', 'target', 'Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)
	test.drop(['team1', 'team2', 'team_id_x', 'team_id_y', 'team_name_x', 'team_name_y','Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)

	return train, train_labels, val, val_labels, val_inv, val_inv_labels, test

