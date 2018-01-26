import sys
import pandas as pd
import numpy as np
import os.path
import time
from sklearn.preprocessing import LabelEncoder

def load(m_params, stage_1):
	train = pd.read_csv("../input/TourneyDetailedResults.csv")
	train_aggs = pd.read_csv("../input/agg_features.csv")
	seeds = pd.read_csv("../input/TourneySeeds.csv")
	team_conference_geog = pd.read_csv("../input/team_conf_and_geog.csv")
	conference_strengths = pd.read_csv("../input/conference_strengths.csv")
	massey_ratings = pd.read_csv("../input/massey_ratings.csv")

	team_conference_strengths = pd.merge(team_conference_geog, conference_strengths, on="conf")[["team_id", "conf_strength"]]

	lbl = LabelEncoder()
	lbl.fit(train_aggs["conf"])
	train_aggs["conf"] = lbl.transform(train_aggs["conf"])

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

	train = pd.merge(train, train_aggs, left_on=["season","team1"], right_on=["season","team_id"])
	train = pd.merge(train, train_aggs, left_on=["season","team2"], right_on=["season","team_id"])
	train = pd.merge(train, team_conference_strengths, left_on="team1", right_on="team_id")
	train = pd.merge(train, team_conference_strengths, left_on="team2", right_on="team_id")
	train = pd.merge(train, massey_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	train = pd.merge(train, massey_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	train["ppg_diff"] = train["ppg_x"] - train["ppg_y"]
	train["mar_diff"] = train["mar_x"] - train["mar_y"]
	train["ppg_nc_diff"] = train["ppg_nc_x"] - train["ppg_nc_y"]
	train["ppgo_nc_diff"] = train["ppgo_nc_x"] - train["ppgo_nc_y"]
	train["mar_nc_diff"] = train["mar_nc_x"] - train["mar_nc_y"]
	train["idx_diff"] = train["idx_x"] - train["idx_y"]
	train["idx_nc_diff"] = train["idx_nc_x"] - train["idx_nc_y"]
	train["tempo_diff"] = train["tempo_x"] - train["tempo_y"]
	train["eff_off_diff"] = train["eff_off_x"] - train["eff_off_y"]
	train["eff_def_diff"] = train["eff_def_x"] - train["eff_def_y"]
	train["3pc_diff"] = train["3pc_x"] - train["3pc_y"]
	train["2pc_diff"] = train["2pc_x"] - train["2pc_y"]
	train["ftpc_diff"] = train["ftpc_x"] - train["ftpc_y"]
	train["blk_diff"] = train["blk_x"] - train["blk_y"]
	train["dreb_diff"] = train["dreb_x"] - train["dreb_y"]
	train["oreb_diff"] = train["oreb_x"] - train["oreb_y"]
	train["tnov_diff"] = train["tnov_x"] - train["tnov_y"]
	train["pf_diff"] = train["pf_x"] - train["pf_y"]
	train["asst_diff"] = train["asst_x"] - train["asst_y"]
	train["attendance_diff"] = train["attendance_x"] - train["attendance_y"]
	train["conf_strength_diff"] = train["conf_strength_x"] - train["conf_strength_y"]
	train["strength_diff"] = train["strength_x"] - train["strength_y"]

	train = train[["target","ppg_diff","mar_diff","ppg_nc_diff","ppgo_nc_diff","mar_nc_diff","idx_diff","idx_nc_diff","tempo_diff","eff_off_diff","eff_def_diff","3pc_diff","2pc_diff","ftpc_diff","blk_diff","dreb_diff","oreb_diff","tnov_diff","pf_diff","asst_diff","attendance_diff","conf_strength_diff","strength_diff"]]

	ix_halfway = np.int(np.floor(len(val)/2))
	val_1 = val[:ix_halfway]
	val_1.columns = ["season","daynum","team1","team1scr","team2","team2scr"]
	val_1["target"] = val_1["team1scr"] - val_1["team2scr"]
	val_2 = val[ix_halfway:]
	val_2.columns = ["season","daynum","team2","team2scr","team1","team1scr"]
	val_2["target"] = val_2["team1scr"] - val_2["team2scr"]
	val = pd.concat([val_1, val_2], axis=0)

	val = pd.merge(val, train_aggs, left_on=["season","team1"], right_on=["season","team_id"])
	val = pd.merge(val, train_aggs, left_on=["season","team2"], right_on=["season","team_id"])
	val = pd.merge(val, team_conference_strengths, left_on="team1", right_on="team_id")
	val = pd.merge(val, team_conference_strengths, left_on="team2", right_on="team_id")
	val = pd.merge(val, massey_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	val = pd.merge(val, massey_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	val["ppg_diff"] = val["ppg_x"] - val["ppg_y"]
	val["mar_diff"] = val["mar_x"] - val["mar_y"]
	val["ppg_nc_diff"] = val["ppg_nc_x"] - val["ppg_nc_y"]
	val["ppgo_nc_diff"] = val["ppgo_nc_x"] - val["ppgo_nc_y"]
	val["mar_nc_diff"] = val["mar_nc_x"] - val["mar_nc_y"]
	val["idx_diff"] = val["idx_x"] - val["idx_y"]
	val["idx_nc_diff"] = val["idx_nc_x"] - val["idx_nc_y"]
	val["tempo_diff"] = val["tempo_x"] - val["tempo_y"]
	val["eff_off_diff"] = val["eff_off_x"] - val["eff_off_y"]
	val["eff_def_diff"] = val["eff_def_x"] - val["eff_def_y"]
	val["3pc_diff"] = val["3pc_x"] - val["3pc_y"]
	val["2pc_diff"] = val["2pc_x"] - val["2pc_y"]
	val["ftpc_diff"] = val["ftpc_x"] - val["ftpc_y"]
	val["blk_diff"] = val["blk_x"] - val["blk_y"]
	val["dreb_diff"] = val["dreb_x"] - val["dreb_y"]
	val["oreb_diff"] = val["oreb_x"] - val["oreb_y"]
	val["tnov_diff"] = val["tnov_x"] - val["tnov_y"]
	val["pf_diff"] = val["pf_x"] - val["pf_y"]
	val["asst_diff"] = val["asst_x"] - val["asst_y"]
	val["attendance_diff"] = val["attendance_x"] - val["attendance_y"]
	val["conf_strength_diff"] = val["conf_strength_x"] - val["conf_strength_y"]
	val["strength_diff"] = val["strength_x"] - val["strength_y"]

	val = val[["target","ppg_diff","mar_diff","ppg_nc_diff","ppgo_nc_diff","mar_nc_diff","idx_diff","idx_nc_diff","tempo_diff","eff_off_diff","eff_def_diff","3pc_diff","2pc_diff","ftpc_diff","blk_diff","dreb_diff","oreb_diff","tnov_diff","pf_diff","asst_diff","attendance_diff","conf_strength_diff","strength_diff"]]

	test = pd.merge(test, train_aggs, left_on=["season","team1"], right_on=["season","team_id"])
	test = pd.merge(test, train_aggs, left_on=["season","team2"], right_on=["season","team_id"])
	test = pd.merge(test, team_conference_strengths, left_on="team1", right_on="team_id")
	test = pd.merge(test, team_conference_strengths, left_on="team2", right_on="team_id")
	test = pd.merge(test, massey_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	test = pd.merge(test, massey_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	test["ppg_diff"] = test["ppg_x"] - test["ppg_y"]
	test["mar_diff"] = test["mar_x"] - test["mar_y"]
	test["ppg_nc_diff"] = test["ppg_nc_x"] - test["ppg_nc_y"]
	test["ppgo_nc_diff"] = test["ppgo_nc_x"] - test["ppgo_nc_y"]
	test["mar_nc_diff"] = test["mar_nc_x"] - test["mar_nc_y"]
	test["idx_diff"] = test["idx_x"] - test["idx_y"]
	test["idx_nc_diff"] = test["idx_nc_x"] - test["idx_nc_y"]
	test["tempo_diff"] = test["tempo_x"] - test["tempo_y"]
	test["eff_off_diff"] = test["eff_off_x"] - test["eff_off_y"]
	test["eff_def_diff"] = test["eff_def_x"] - test["eff_def_y"]
	test["3pc_diff"] = test["3pc_x"] - test["3pc_y"]
	test["2pc_diff"] = test["2pc_x"] - test["2pc_y"]
	test["ftpc_diff"] = test["ftpc_x"] - test["ftpc_y"]
	test["blk_diff"] = test["blk_x"] - test["blk_y"]
	test["dreb_diff"] = test["dreb_x"] - test["dreb_y"]
	test["oreb_diff"] = test["oreb_x"] - test["oreb_y"]
	test["tnov_diff"] = test["tnov_x"] - test["tnov_y"]
	test["pf_diff"] = test["pf_x"] - test["pf_y"]
	test["asst_diff"] = test["asst_x"] - test["asst_y"]
	test["attendance_diff"] = test["attendance_x"] - test["attendance_y"]
	test["conf_strength_diff"] = test["conf_strength_x"] - test["conf_strength_y"]
	test["strength_diff"] = test["strength_x"] - test["strength_y"]

	test.sort(['season', 'team1', 'team2'], inplace=True)

	test = test[["ppg_diff","mar_diff","ppg_nc_diff","ppgo_nc_diff","mar_nc_diff","idx_diff","idx_nc_diff","tempo_diff","eff_off_diff","eff_def_diff","3pc_diff","2pc_diff","ftpc_diff","blk_diff","dreb_diff","oreb_diff","tnov_diff","pf_diff","asst_diff","attendance_diff","conf_strength_diff","strength_diff"]]

	#print(len(train), len(val))

	train_labels = train['target']
	val_labels = val['target']

	train.drop(['target'], axis=1, inplace=True)
	val.drop(['target'], axis=1, inplace=True)
	#test.drop(['team_name_x', 'team_name_y','Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)

	return train, train_labels, val, val_labels, test
