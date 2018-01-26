import sys
import pandas as pd
import numpy as np
import scipy as sp
import data_pointspreads as data
import argparse
import pickle as pkl
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor, RandomizedLasso
from sklearn.metrics import log_loss
from scipy.stats import norm

parser = argparse.ArgumentParser(description='LR NCAA Basketball Model')
parser.add_argument('-sy','--start_year', type=int, default=2013)
parser.add_argument('-ey','--end_year', type=int, default=2016)

m_params = vars(parser.parse_args())
if m_params["start_year"] < m_params["end_year"]:
	stage_1 = True
	stage_2 = False
else:
	stage_2 = True
	stage_1 = False

X, y, X_val, y_val, X_val2, y_val2, X_sub = data.load(m_params, stage_1)
vars = X.columns

print("NCAA: L1 classification...\n") 
clf = Lasso(alpha=0.05)

pred_matrix_val = np.zeros((len(X_val), 100))
pred_matrix_test = np.zeros((len(X_sub), 100))
for i in range(100):
	print(i)
	# Load data
	X, y, X_val, y_val, X_val2, y_val2, X_sub = data.load(m_params, stage_1)

	scaler = StandardScaler()
	X_sub = scaler.fit_transform(X_sub)
	if stage_1:
		X_val = scaler.transform(X_val)
		X_val2 = scaler.transform(X_val2)
	X = scaler.fit_transform(X)

	clf.fit(X,y)
	print(pd.DataFrame({'Columns': vars, 'Coef': clf.coef_}))
	if stage_1:
		pred_val1 = clf.predict(X_val)
		pred_val2 = clf.predict(X_val2)
		pred_matrix_val[:,i] = (pred_val1 - pred_val2) / 2
		print(-np.sum(np.log(norm.cdf(pred_matrix_val[:,i] / 12.1)))/252)
	pred_test = clf.predict(X_sub)
	pred_matrix_test[:,i] = pred_test
pred_medians_val = norm.cdf(np.median(pred_matrix_val, axis=1) / 12.1)
pred_medians_test = norm.cdf(np.median(pred_matrix_test, axis=1) / 12.1)

if stage_1:
	season_scores = []
	for i in range(4):
		ll = -np.sum(np.log(pred_medians_val[i*63:(i+1)*63]))/63
		print(ll)
		season_scores.append(ll)
	print('total', np.mean(season_scores))

print("Saving Results.")
if stage_1:
	preds = pd.read_csv("../input/sample_submission_stage1.csv")
	preds["pred"] = pred_medians_test
	preds.to_csv("../output/" + 'lr_stage1' + '.csv', index=False)
else:
	preds = pd.read_csv("../input/sample_submission_stage2.csv")
	preds["pred"] = pred_medians_test
	preds.to_csv("../output/" + 'lr_stage2' + '.csv', index=False)
