import sys
import pandas as pd
import numpy as np
import os.path
import time
from sklearn.preprocessing import LabelEncoder

def load(year, stage_1):
	train = pd.read_csv("../input/RegularSeasonDetailedResults.csv")
	val = pd.read_csv("../input/TourneyDetailedResults.csv")

	train = train[["Season", "Daynum", "Wteam", "Wscore", "Lteam", "Lscore", "Wloc"]]
	val = val[["Season", "Daynum", "Wteam", "Wscore", "Lteam", "Lscore", "Wloc"]]

	train = pd.concat([train, val], axis=0)

	'''
	The Dixon Coles model requires the data to be in the format of hometeam, awayteam, homescore, awayscore 
	to take account of hometeam bias parameter of the model.
	'''

	train_homewins = train[train["Wloc"] == "H"]
	train_awaywins = train[train["Wloc"] == "A"]
	train_other = train[train["Wloc"] == "N"]

	train_homewins["Hteam"] = train_homewins["Wteam"]
	train_homewins["Ateam"] = train_homewins["Lteam"]
	train_homewins["Hscore"] = train_homewins["Wscore"]
	train_homewins["Ascore"] = train_homewins["Lscore"]

	train_awaywins["Hteam"] = train_awaywins["Lteam"]
	train_awaywins["Ateam"] = train_awaywins["Wteam"]
	train_awaywins["Hscore"] = train_awaywins["Lscore"]
	train_awaywins["Ascore"] = train_awaywins["Wscore"]

	train_other["Hteam"] = np.where(train_other["Wteam"]>train_other["Lteam"], train_other["Wteam"], train_other["Lteam"])
	train_other["Ateam"] = np.where(train_other["Wteam"]>train_other["Lteam"], train_other["Lteam"], train_other["Wteam"])
	train_other["Hscore"] = np.where(train_other["Wteam"]>train_other["Lteam"], train_other["Wscore"], train_other["Lscore"])
	train_other["Ascore"] = np.where(train_other["Wteam"]>train_other["Lteam"], train_other["Lscore"], train_other["Wscore"])

	train = pd.concat([train_homewins, train_awaywins, train_other], axis=0)
	train = train[["Season", "Daynum", "Hteam", "Ateam", "Hscore", "Ascore", "Wloc"]]

	train["Hteam"] = train["Hteam"] - 1101
	train["Ateam"] = train["Ateam"] - 1101

	train.sort(["Season", "Daynum", "Hteam"])

	'''
	Geographical lat and lon for both teams are useful to determine how far the teams have to travel to the game.
	This can be added to the Dixon Coles model. Currently lacking lat/long for the neutral venues.
	'''
	team_conf_geog = pd.read_csv("../input/team_conf_and_geog.csv")
	team_conf_geog["team_id"] -= 1101
	team_conf = team_conf_geog[["team_id", "conf"]]

	train = pd.merge(train, team_conf_geog, left_on=["Hteam"], right_on=["team_id"])
	train = pd.merge(train, team_conf_geog, left_on=["Ateam"], right_on=["team_id"])

	train["Distance"] = np.round(np.arccos(np.sin(train["lat_x"]*3.1416/180)*np.sin(train["lat_y"]*3.1416/180) + np.cos(train["lat_x"]*3.1416/180)*np.cos(train["lat_y"]*3.1416/180)*np.cos(train["lng_y"]*3.1416/180-train["lng_x"]*3.1416/180) ) * 6.371,2)

	train = train[["Season", "Daynum", "Hteam", "Ateam", "Hscore", "Ascore", "Wloc", "Distance"]]

	'''
	We can split the training data into two parts, one to run the model with, and the other for validating.
	'''

	train = train[train["Season"] == year]

	val = train[train["Daynum"] >= 136]
	val.sort_values(["Season", "Daynum", "Hteam"], inplace=True)
	train = train[train["Daynum"] < 136]

	if stage_1:
		test = pd.read_csv("../input/sample_submission_stage1.csv", sep="[,_]").reset_index().ix[:,:3]
	else:
		test = pd.read_csv("../input/sample_submission_stage2.csv", sep="[,_]").reset_index().ix[:,:3]

	test.columns = ["Season", "Team1", "Team2"]
	test = test[test["Season"] == year]
	test["Team1"] = test["Team1"] - 1101
	test["Team2"] = test["Team2"] - 1101

	teams = pd.read_csv("../input/Teams.csv")
	teams["Team_Id"] = teams["Team_Id"] - 1101

	print(len(train), len(val), len(test))

	return train.values, val.values, test.values, teams
