import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import data_pointspreads as data
import argparse
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from scipy.stats import norm

parser = argparse.ArgumentParser(description='XGBoost for Springleaf')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=100)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=300)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=1)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_val, y_val, X_sub = data.load(m_params)

print("NCAA Machine Learning Mania 2016-17: binary classification...\n")
model_name = "_".join(['xgb_pointspread',str(m_params['n_rounds']),str(m_params['n_features']),str(m_params['eta'])])

xgb_param = {'silent' : 1, 'max_depth' : 6, 'eta': m_params['eta'], 'objective':'reg:linear', 'min_child_weight':6, 'colsample_bytree':0.5, 'subsample':0.7}

if m_params['cv']:
	# do cross validation scoring
	kf = StratifiedShuffleSplit(y, n_iter=4, test_size=0.25, random_state=m_params['r_seed'])
	scr = np.zeros([len(kf)])
	for i, (tr_ix, val_ix) in enumerate(kf):
		dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix])
		dval = xgb.DMatrix(X[val_ix], y[val_ix])
		clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dval,'val']))
		pred = clf.predict(dval)
		scr[i] = log_loss(y[val_ix], pred)
		print(scr[i])
	print np.mean(scr)

else:
	pred_matrix_val = np.zeros((len(X_val), 10))
	pred_matrix_test = np.zeros((len(X_sub), 10))
	for i in range(10):
		X, y, X_val, y_val, X_sub = data.load(m_params)
		dtrain = xgb.DMatrix(X, y)
		dval = xgb.DMatrix(X_val)
		dtest = xgb.DMatrix(X_sub)
		clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'])
		pred_val = clf.predict(dval)
		pred_test = clf.predict(dtest)
		pred_matrix_val[:,i] = pred_val
		pred_matrix_test[:,i] = pred_test
	pred_means_val = norm.cdf(np.mean(pred_matrix_val, axis=1) / 12.1)
	pred_means_test = norm.cdf(np.mean(pred_matrix_test, axis=1) / 12.1)

	y_val = (y_val > 0)
	print(log_loss(y_val, pred_means_val))

	val = pd.read_csv("../input/TourneyCompactResults.csv")
	val = val[val["Season"] >= 2013]
	val["pred"] = pred_means_val
	print(val)

	print("Saving Results.")
	preds = pd.read_csv("../input/sample_submission.csv")
	preds["pred"] = pred_means_test
	preds.to_csv("../output/" + model_name + '.csv', index=False)
