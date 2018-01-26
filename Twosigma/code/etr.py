import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data2 as data
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1200)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=2000)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='std')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)

print("Two Sigma: Extra Trees Classifier...\n") 
clf =  ExtraTreesRegressor(n_estimators=500, verbose=1, max_features=65, criterion= 'mse', min_samples_split= 5, max_depth=17, n_jobs=-1, min_samples_leaf=2)

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros(X.shape[0])
sub_pred = np.zeros(X_sub.shape[0])
for i, (tr_ix, val_ix) in enumerate(kf):
	clf.fit(X[tr_ix], y[tr_ix])
	pred = clf.predict(X[val_ix])
	oob_pred[val_ix] = np.array(pred)
	sub_pred += clf.predict(X_sub) / 5
	scr[i] = mean_squared_error(y[val_ix], np.array(pred))
	print('Train score is:', scr[i])
print(mean_squared_error(y, oob_pred))

oob_pred_filename = '../output/oob_pred_etreg_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_etreg_' + str(np.mean(scr))
pkl.dump(oob_pred[:, None], open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred[:, None], open(sub_pred_filename + '.p', 'wb'))
