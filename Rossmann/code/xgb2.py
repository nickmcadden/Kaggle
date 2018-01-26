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
import data
import argparse
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=100)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=200)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.35)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=4)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
m_params = vars(parser.parse_args())

def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    # y = y.values
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe

# Load data
X, y, X_sub, ids = data.load(m_params)
X, y = shuffle(X, y, random_state = m_params['r_seed'])

print("Springleaf: binary classification...\n")
model_name = "_".join(['xgb',str(m_params['n_rounds']),str(m_params['n_features']),str(m_params['eta'])])

xgb_param = {'silent' : 1, 'max_depth' : 9, 'eta': m_params['eta'], 'objective':'reg:linear', 'min_child_weight':5, 'colsample_bytree':0.75, 'subsample':0.8}

if m_params['cv']:
	# do cross validation scoring
	kf = KFold(n=len(y), n_folds=4, shuffle=True, random_state=m_params['r_seed'])
	scr = np.zeros([len(kf)])
	for i, (tr_ix, val_ix) in enumerate(kf):
		dtrain = xgb.DMatrix(X[tr_ix], np.log(y[tr_ix]+1))
		dval = xgb.DMatrix(X[val_ix], np.log(y[val_ix]+1))
		clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dval,'val']), feval=rmspe_xg)
		pred = clf.predict(dval)
		scr[i] = roc_auc_score(y[val_ix], pred)
		print(scr[i])
	print np.mean(scr)

else:
	# Train on full data
	dtrain = xgb.DMatrix(X,y)
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], feval=rmspe)
	clf.save_model('output/' + model_name + '.model')
	pred = clf.predict(dtest)

	print("Saving Results.")
	preds = pd.DataFrame({"ID": ids, "target": pred})
	preds.to_csv('output/' + model_name + '.csv', index=False)
