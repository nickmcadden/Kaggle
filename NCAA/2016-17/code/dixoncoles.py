import sys
import pandas as pd
import numpy as np
import time
import data_dixoncoles as data
import argparse
from sklearn.metrics import log_loss
from scipy.stats import norm
from scipy.special import factorial
from scipy.optimize import minimize
from scipy.stats import skellam

parser = argparse.ArgumentParser(description='Dixon Coles Model')
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=1)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-sy','--start_year', type=int, default=2017)
parser.add_argument('-ey','--end_year', type=int, default=2017)

m_params = vars(parser.parse_args())
if m_params["start_year"] < m_params["end_year"]:
	stage_1 = True
	stage_2 = False
else:
	stage_2 = True
	stage_1 = False

print("NCAA Machine Learning Mania 2016-17: MLE optimisation via Dixon-Coles method...\n")

def oddspredict(fixtures, att_params, def_params, hmean, amean):
	resultodds = []
	neutralscore = (hmean+amean)/2
	for j in range(len(fixtures)):
		lamda = neutralscore * att_params[fixtures[j,0]] * def_params[fixtures[j,1]]
		mu = neutralscore * att_params[fixtures[j,1]] * def_params[fixtures[j,0]]
		px = skellam.cdf(-1, lamda, mu)
		p0 = skellam.pmf(0, lamda, mu)
		resultodds.append(1-(px+p0*0.5))
	return np.array(resultodds)

def get_vec(teams, params):
	vec = np.array([params[team] for team in teams])
	return vec

'''
def objective(params, hmean, amean):
	attparams = params[:364]
	defparams = params[364:]
	f=0
	for i in range(len(X)):
		x = X[i,4] # home score
		y = X[i,5] # away score
		h = X[i,2] # home team
		a = X[i,3] # away team
		lamda = hmean * attparams[h] * defparams[a]
		mu = amean * attparams[a] * defparams[h]
		p = ((np.power(lamda,x)*np.exp(-lamda)) / factorial(x, exact=False)) * ((np.power(mu,y)*np.exp(-mu)) / factorial(y, exact=False))
		f -= np.log(p)
	return f
'''

def objective_vectorized(params, hmean, amean):
	# attack and defense params
	attparams = params[:364]
	defparams = params[364:728]
	# distance coefficient
	#dcf = params[728]
	home_teams = X[:,2]
	away_teams = X[:,3]
	home_team_scores = X[:,4]
	away_team_scores = X[:,5]
	#travel_distances = X[:,7].astype(np.float32)
	ht_att_vec = get_vec(home_teams, attparams)
	ht_def_vec = get_vec(home_teams, defparams)
	at_att_vec = get_vec(away_teams, attparams)
	at_def_vec = get_vec(away_teams, defparams)
	lamda = hmean * ht_att_vec * at_def_vec
	mu = amean * at_att_vec * ht_def_vec # - (travel_distances * dcf)
	p = np.sum(lamda) + np.sum(mu) - np.sum(home_team_scores*np.log(lamda)) - np.sum(away_team_scores*np.log(mu))
	return p

submission_probs = []

for year in range(m_params['start_year'], m_params['end_year']+1):
	print("year:", year)
	# Load data
	X, X_val, X_sub, Teams = data.load(year, stage_1)

	initparams = np.ones(728).astype(np.float32)
	#X=X[:1000]

	'''
	neutralgames = (X[:,6]=='N')
	meanhomescore = np.sum(X[:,4].astype(np.float32) * ~neutralgames) / np.sum(~neutralgames)
	meanawayscore = np.sum(X[:,5].astype(np.float32) * ~neutralgames) / np.sum(~neutralgames)
	meanneutralscore = (np.sum(X[:,5] * neutralgames) / np.sum(neutralgames) + np.sum(X[:,4] * neutralgames) / np.sum(neutralgames)) / 2
	meanhomescore_vec = meanhomescore * ~neutralgames + meanneutralscore * neutralgames
	meanawayscore_vec = meanawayscore * ~neutralgames + meanneutralscore * neutralgames

	print(meanneutralscore, meanhomescore, meanawayscore)
	'''

	meanhomescore = np.mean(X[:,4])
	meanawayscore = np.mean(X[:,5])
	meanhomescore_vec = np.array([(meanhomescore+meanawayscore)/2] * len(X)) + 2
	meanawayscore_vec = np.array([(meanhomescore+meanawayscore)/2] * len(X))

	print("Optimising attack and defense parameters")
	t0 = time.time()
	optim = minimize(objective_vectorized, initparams, args=(meanhomescore_vec, meanawayscore_vec), method="Powell")
	t1 = time.time()
	print(t1-t0, "seconds")

	attparams = optim['x'][:364]
	defparams = optim['x'][364:728]

	attparams_df = pd.DataFrame({'teamid': range(364), 'attack': attparams})
	defparams_df = pd.DataFrame({'teamid': range(364), 'defence': defparams})

	Teams = pd.merge(Teams, attparams_df, left_on=["Team_Id"], right_on=["teamid"])
	Teams = pd.merge(Teams, defparams_df, left_on=["Team_Id"], right_on=["teamid"])
	Teams['strength'] = Teams['attack'] / Teams['defence']
	Teams = Teams.sort('strength', ascending=False)
	print(Teams.ix[:, ['Team_Id','Team_Name','attack','defence','strength']])

	if stage_1:
		print("Predicting odds based on optimised parameters")
		# Get odds for the cv tournament data to score against the log loss measure
		fixtures = X_val[:,2:4]
		probs = oddspredict(fixtures, attparams, defparams, meanhomescore, meanawayscore)
		X_val = np.concatenate((X_val, np.round(probs[:, None] ,2)), axis=1)
		print(X_val)
		y_val = np.array(X_val[:,4] > X_val[:,5])
		print("logloss", log_loss(y_val, probs))

	# Get odds for all potential matchups for this year, for the submission file
	fixtures = X_sub[:,1:3]
	probs = oddspredict(fixtures, attparams, defparams, meanhomescore, meanawayscore)
	submission_probs.extend(probs)

print("Saving Results.")
if stage_1:
	preds = pd.read_csv("../input/sample_submission_stage1.csv")
	preds["pred"] = submission_probs
	preds.to_csv("../output/dixoncoles_stage1" + '.csv', index=False)
else:
	preds = pd.read_csv("../input/sample_submission_stage2.csv")
	preds["pred"] = submission_probs
	preds.to_csv("../output/dixoncoles_stage2" + '.csv', index=False)


