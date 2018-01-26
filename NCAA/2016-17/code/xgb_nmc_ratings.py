import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import data_nmc_ratings as data
import argparse
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from scipy.stats import norm

parser = argparse.ArgumentParser(description='LR NCAA Basketball Model')
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=300)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
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

print("NCAA Machine Learning Mania 2016-17: binary classification...\n")
model_name = 'xgb_nmc_ratings'

xgb_param = {'silent' : 1, 'max_depth' : 6, 'eta': m_params['eta'], 'objective':'reg:linear', 'min_child_weight':6, 'colsample_bytree':0.5, 'subsample':0.7}

pred_matrix_val = np.zeros((len(X_val), 10))
pred_matrix_test = np.zeros((len(X_sub), 10))
for i in range(10):
	X, y, X_val, y_val, X_val2, y_val2, X_sub = data.load(m_params, stage_1)

	dtrain = xgb.DMatrix(X, y)
	dval = xgb.DMatrix(X_val)
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'])
	pred_val = clf.predict(dval)
	pred_test = clf.predict(dtest)
	pred_matrix_val[:,i] = pred_val
	pred_matrix_test[:,i] = pred_test
	print(-np.sum(np.log(norm.cdf(pred_matrix_val[:,i] / 12.1)))/252)

pred_medians_val = norm.cdf(np.median(pred_matrix_val, axis=1) / 12.1)
pred_medians_test = norm.cdf(np.median(pred_matrix_test, axis=1) / 12.1)

if stage_1:
	y_val = (y_val > 0).astype(int)
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
	preds.to_csv("../output/" + 'xgb_nmc_ratings_stage1' + '.csv', index=False)
else:
	preds = pd.read_csv("../input/sample_submission_stage2.csv")
	preds["pred"] = pred_medians_test
	preds.to_csv("../output/" + 'xgb_nmc_ratings_stage2' + '.csv', index=False)
