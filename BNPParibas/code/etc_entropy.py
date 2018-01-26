import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data
import argparse
import pickle as pkl
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.utils import shuffle

def log_loss(act, pred):
    """ Vectorised computation of logloss """
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1000)
parser.add_argument('-n','--n_rounds', help='Number of iterations', type=int, default=1250)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='std')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)

print("BNP Parabas: classification...\n") 
clf = ExtraTreesClassifier(n_estimators=m_params['n_rounds'], max_features= 50, criterion= 'entropy', min_samples_split= 4, max_depth= 35, min_samples_leaf= 2, n_jobs = -1)      

if m_params['cv']:
	# do cross validation scoring
	kf = KFold(X.shape[0], n_folds=4, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros(X.shape[0])
	sub_pred = np.zeros((X_sub.shape[0], 4))
	for i, (tr_ix, val_ix) in enumerate(kf):
		clf.fit(X[tr_ix], y[tr_ix])
		pred = clf.predict_proba(X[val_ix])
		oob_pred[val_ix] = np.array(pred[:,1])
		sub_pred[:,i] = clf.predict_proba(X_sub)[:,1]
		scr[i] = log_loss(y[val_ix], np.array(pred[:,1]))
		print('Train score is:', scr[i])
	print(log_loss(y, oob_pred))
	print oob_pred[1:10]
	sub_pred = sub_pred.mean(axis=1)
	oob_pred_filename = '../output/oob_pred_etcentropy_' + str(np.mean(scr))
	sub_pred_filename = '../output/sub_pred_etcentropy_' + str(np.mean(scr))
	pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
	pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
	preds = pd.DataFrame({"ID": ids, "PredictedProb": sub_pred})
	preds.to_csv(sub_pred_filename + '.csv', index=False)

else:
	X, y = shuffle(X, y)
	# Train on full data
	print("Training on full data")
	clf.fit(X,y)

	print("Creating predictions")
	pred = clf.predict_proba(X_sub)

	print("Saving Results.")
	model_name = '../output/pred_etc_' + str(m_params['n_rounds'])
	preds = pd.DataFrame({"ID": ids, "PredictedProb": pred[:,1]})
	preds.to_csv(model_name + '.csv', index=False)
