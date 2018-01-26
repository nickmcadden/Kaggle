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
	train_aggs = pd.read_csv("../input/agg_features.csv")
	seeds = pd.read_csv("../input/TourneySeeds.csv")
	team_conference_geog = pd.read_csv("../input/team_conf_and_geog.csv")
	conference_strengths = pd.read_csv("../input/conference_strengths.csv")
	elo_strength_ratings = pd.read_csv("../input/elo_strength_ratings.csv")

	team_conference_strengths = pd.merge(team_conference_geog, conference_strengths, on="conf")[["team_id", "conf_strength"]]

	lbl = LabelEncoder()
	lbl.fit(train_aggs["conf"])
	train_aggs["conf"] = lbl.transform(train_aggs["conf"])

	if stage_1:
		test = pd.read_csv("../input/sample_submission_stage1.csv", sep="[,_]").reset_index().ix[:,:3]
	else:
		test = pd.read_csv("../input/sample_submission_stage2.csv", sep="[,_]").reset_index().ix[:,:3]
	test.columns = ["season", "team1", "team2"]

	daytoround = {134:0,135:0,136:1,137:1,138:2,139:2,143:3,144:3,145:4,146:4,152:5,154:6}

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
	#train = pd.merge(train, seeds, left_on=["season","team1"], right_on=["Season","Team"])
	#train = pd.merge(train, seeds, left_on=["season","team2"], right_on=["Season","Team"])
	train = pd.merge(train, team_conference_strengths, left_on="team1", right_on="team_id")
	train = pd.merge(train, team_conference_strengths, left_on="team2", right_on="team_id")
	train = pd.merge(train, elo_strength_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	train = pd.merge(train, elo_strength_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	train["round"] = get_rounds_vec(list(train["daynum"]), daytoround)
	#train["Seed_Diff"] = train["Seed_x"] - train["Seed_y"]
	train["ppg_diff"] = train["ppg_x"] - train["ppg_y"]
	train["mar_diff"] = train["mar_x"] - train["mar_y"]
	train["winperc_diff"] = train["winperc_x"] - train["winperc_y"]
	train["ppg_nc_diff"] = train["ppg_nc_x"] - train["ppg_nc_y"]
	train["ppgo_nc_diff"] = train["ppgo_nc_x"] - train["ppgo_nc_y"]
	train["mar_nc_diff"] = train["mar_nc_x"] - train["mar_nc_y"]
	train["winperc_nc_diff"] = train["winperc_nc_x"] - train["winperc_nc_y"]
	train["idx_diff"] = train["idx_x"] - train["idx_y"]
	train["idx_nc_diff"] = train["idx_nc_x"] - train["idx_nc_y"]
	train["tempo_diff"] = train["tempo_x"] - train["tempo_y"]
	train["eff_off_diff"] = train["eff_off_x"] - train["eff_def_y"]
	train["eff_def_diff"] = train["eff_def_x"] - train["eff_off_y"]
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

	val.columns = ["season","daynum","team1","team1scr","team2","team2scr"]
	val["target"] = val["team1scr"] - val["team2scr"]

	val = pd.merge(val, train_aggs, left_on=["season","team1"], right_on=["season","team_id"])
	val = pd.merge(val, train_aggs, left_on=["season","team2"], right_on=["season","team_id"])
	#val = pd.merge(val, seeds, left_on=["season","team1"], right_on=["Season","Team"])
	#val = pd.merge(val, seeds, left_on=["season","team2"], right_on=["Season","Team"])
	val = pd.merge(val, team_conference_strengths, left_on="team1", right_on="team_id")
	val = pd.merge(val, team_conference_strengths, left_on="team2", right_on="team_id")
	val = pd.merge(val, elo_strength_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	val = pd.merge(val, elo_strength_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	val["round"] = get_rounds_vec(list(val["daynum"]), daytoround)
	#val["Seed_Diff"] = val["Seed_x"] - val["Seed_y"]
	val["ppg_diff"] = val["ppg_x"] - val["ppg_y"]
	val["mar_diff"] = val["mar_x"] - val["mar_y"]
	val["winperc_diff"] = val["winperc_x"] - val["winperc_y"]
	val["ppg_nc_diff"] = val["ppg_nc_x"] - val["ppg_nc_y"]
	val["ppgo_nc_diff"] = val["ppgo_nc_x"] - val["ppgo_nc_y"]
	val["mar_nc_diff"] = val["mar_nc_x"] - val["mar_nc_y"]
	val["winperc_nc_diff"] = val["winperc_nc_x"] - val["winperc_nc_y"]
	val["idx_diff"] = val["idx_x"] - val["idx_y"]
	val["idx_nc_diff"] = val["idx_nc_x"] - val["idx_nc_y"]
	val["tempo_diff"] = val["tempo_x"] - val["tempo_y"]
	val["eff_off_diff"] = val["eff_off_x"] - val["eff_def_y"]
	val["eff_def_diff"] = val["eff_def_x"] - val["eff_off_y"]
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

	val_inv = val.copy()
	val_inv = val_inv[["season","daynum","team1","team1scr","team2","team2scr"]]
	val_inv["team1"] = val["team2"]
	val_inv["team2"] = val["team1"]
	val_inv["team1scr"] = val["team2scr"]
	val_inv["team2scr"] = val["team1scr"]
	val_inv["target"] = val_inv["team1scr"] - val_inv["team2scr"]

	val_inv = pd.merge(val_inv, train_aggs, left_on=["season","team1"], right_on=["season","team_id"])
	val_inv = pd.merge(val_inv, train_aggs, left_on=["season","team2"], right_on=["season","team_id"])
	#val = pd.merge(val, seeds, left_on=["season","team1"], right_on=["Season","Team"])
	#val = pd.merge(val, seeds, left_on=["season","team2"], right_on=["Season","Team"])
	val_inv = pd.merge(val_inv, team_conference_strengths, left_on="team1", right_on="team_id")
	val_inv = pd.merge(val_inv, team_conference_strengths, left_on="team2", right_on="team_id")
	val_inv = pd.merge(val_inv, elo_strength_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	val_inv = pd.merge(val_inv, elo_strength_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	val_inv["round"] = get_rounds_vec(list(val_inv["daynum"]), daytoround)
	#val["Seed_Diff"] = val["Seed_x"] - val["Seed_y"]
	val_inv["ppg_diff"] = val_inv["ppg_x"] - val_inv["ppg_y"]
	val_inv["mar_diff"] = val_inv["mar_x"] - val_inv["mar_y"]
	val_inv["winperc_diff"] = val_inv["winperc_x"] - val_inv["winperc_y"]
	val_inv["ppg_nc_diff"] = val_inv["ppg_nc_x"] - val_inv["ppg_nc_y"]
	val_inv["ppgo_nc_diff"] = val_inv["ppgo_nc_x"] - val_inv["ppgo_nc_y"]
	val_inv["mar_nc_diff"] = val_inv["mar_nc_x"] - val_inv["mar_nc_y"]
	val_inv["winperc_nc_diff"] = val_inv["winperc_nc_x"] - val_inv["winperc_nc_y"]
	val_inv["idx_diff"] = val_inv["idx_x"] - val_inv["idx_y"]
	val_inv["idx_nc_diff"] = val_inv["idx_nc_x"] - val_inv["idx_nc_y"]
	val_inv["tempo_diff"] = val_inv["tempo_x"] - val_inv["tempo_y"]
	val_inv["eff_off_diff"] = val_inv["eff_off_x"] - val_inv["eff_def_y"]
	val_inv["eff_def_diff"] = val_inv["eff_def_x"] - val_inv["eff_off_y"]
	val_inv["3pc_diff"] = val_inv["3pc_x"] - val_inv["3pc_y"]
	val_inv["2pc_diff"] = val_inv["2pc_x"] - val_inv["2pc_y"]
	val_inv["ftpc_diff"] = val_inv["ftpc_x"] - val_inv["ftpc_y"]
	val_inv["blk_diff"] = val_inv["blk_x"] - val_inv["blk_y"]
	val_inv["dreb_diff"] = val_inv["dreb_x"] - val_inv["dreb_y"]
	val_inv["oreb_diff"] = val_inv["oreb_x"] - val_inv["oreb_y"]
	val_inv["tnov_diff"] = val_inv["tnov_x"] - val_inv["tnov_y"]
	val_inv["pf_diff"] = val_inv["pf_x"] - val_inv["pf_y"]
	val_inv["asst_diff"] = val_inv["asst_x"] - val_inv["asst_y"]
	val_inv["attendance_diff"] = val_inv["attendance_x"] - val_inv["attendance_y"]
	val_inv["conf_strength_diff"] = val_inv["conf_strength_x"] - val_inv["conf_strength_y"]
	val_inv["strength_diff"] = val_inv["strength_x"] - val_inv["strength_y"]

	test = pd.merge(test, train_aggs, left_on=["season","team1"], right_on=["season","team_id"])
	test = pd.merge(test, train_aggs, left_on=["season","team2"], right_on=["season","team_id"])
	#test = pd.merge(test, seeds, left_on=["season","team1"], right_on=["Season","Team"])
	#test = pd.merge(test, seeds, left_on=["season","team2"], right_on=["Season","Team"])
	test = pd.merge(test, team_conference_strengths, left_on="team1", right_on="team_id")
	test = pd.merge(test, team_conference_strengths, left_on="team2", right_on="team_id")
	test = pd.merge(test, elo_strength_ratings, left_on=["season","team1"], right_on=["season","team_id"])
	test = pd.merge(test, elo_strength_ratings, left_on=["season","team2"], right_on=["season","team_id"])

	test["round"] = 1
	#test["Seed_Diff"] = test["Seed_x"] - test["Seed_y"]
	test["ppg_diff"] = test["ppg_x"] - test["ppg_y"]
	test["mar_diff"] = test["mar_x"] - test["mar_y"]
	test["winperc_diff"] = test["winperc_x"] - test["winperc_y"]
	test["ppg_nc_diff"] = test["ppg_nc_x"] - test["ppg_nc_y"]
	test["ppgo_nc_diff"] = test["ppgo_nc_x"] - test["ppgo_nc_y"]
	test["mar_nc_diff"] = test["mar_nc_x"] - test["mar_nc_y"]
	test["winperc_nc_diff"] = test["winperc_nc_x"] - test["winperc_nc_y"]
	test["idx_diff"] = test["idx_x"] - test["idx_y"]
	test["idx_nc_diff"] = test["idx_nc_x"] - test["idx_nc_y"]
	test["tempo_diff"] = test["tempo_x"] - test["tempo_y"]
	test["eff_off_diff"] = test["eff_off_x"] - test["eff_def_y"]
	test["eff_def_diff"] = test["eff_def_x"] - test["eff_off_y"]
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

	train_labels = train['target']

	val = val.sort_values(by=["season", "daynum", "team1", "team2"]).reset_index()
	val_inv = val_inv.sort_values(by=["season", "daynum", "team2", "team1"]).reset_index()
	val_labels = val['target']
	val_inv_labels = val_inv['target']

	train.drop(['daynum', 'team1', 'team2', 'team_id_x', 'team_id_y', 'team_name_x', 'team_name_y', 'team1scr', 'team2scr', 'target', 'Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)
	val.drop(['index', 'team1', 'team2', 'team_id_x', 'team_id_y', 'daynum', 'team_name_x', 'team_name_y', 'team1scr', 'team2scr', 'target', 'Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)
	val_inv.drop(['index', 'team1', 'team2', 'team_id_x', 'team_id_y', 'daynum', 'team_name_x', 'team_name_y', 'team1scr', 'team2scr', 'target', 'Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)
	test.drop(['team_name_x', 'team1', 'team2', 'team_id_x', 'team_id_y', 'team_name_y','Season_x', 'Season_y', 'Team_x', 'Team_y'], axis=1, inplace=True)

	return train, train_labels, val, val_labels, val_inv, val_inv_labels, test

