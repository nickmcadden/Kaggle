# Try these thiing 
#############################
# check categorical encoding to make sure the correlation type is working correctly
# cap outliers for the numerical columns
# further clean the string based features such as empty string replacement
# tune the XGboost more to accept larger dimensional input
# look at calibration curve of final output and create own adjuster
# Flag up holidays to add to the model.
# bin data to increase linearity

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import data5 as data
import argparse
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import KNNClassifier
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='XGBoost for Springleaf')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=100)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=20)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.1)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=4)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)
X, y = shuffle(X, y, random_state = m_params['r_seed'])

pd.DataFrame(y).to_csv('input/target.csv', index=False)

print("Springleaf: binary classification...\n")
model_name = "_".join(['xgb',str(m_params['n_rounds']),str(m_params['n_features']),str(m_params['eta'])])

xgb_param = {'silent' : 1, 'max_depth' : 9, 'alpha' : 4, 'eta': m_params['eta'], 'objective':'binary:logistic', 'eval_metric':'auc', 'min_child_weight':6, 'colsample_bytree':0.5, 'subsample':0.7}

if m_params['cv']:
	# do cross validation scoring
	kf = StratifiedShuffleSplit(y, n_iter=4, test_size=0.25, random_state=m_params['r_seed'])
	scr = np.zeros([len(kf)])
	for i, (tr_ix, val_ix) in enumerate(kf):
		dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix])
		dval = xgb.DMatrix(X[val_ix], y[val_ix])
		clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dval,'val']))
		pred = clf.predict(dval)
		scr[i] = roc_auc_score(y[val_ix], pred)
		print(scr[i])
	print np.mean(scr)

else:
	# Train on full data
	dtrain = xgb.DMatrix(X,y)
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'])
	clf.save_model('output/' + model_name + '.model')
	pred = clf.predict(dtest)

	print("Saving Results.")
	preds = pd.DataFrame({"ID": ids, "target": pred})
	preds.to_csv('output/' + model_name + '.csv', index=False)
