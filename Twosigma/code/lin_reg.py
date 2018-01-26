import sys
import pandas as pd
import numpy as np
import scipy as sp
import data_linear as data
import argparse
import pickle as pkl
from sklearn.metrics import mean_squared_error
from scipy import stats
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1200)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=2000)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='onehot')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)

X = np.nan_to_num(np.array(X.todense()))
X_sub = np.nan_to_num(np.array(X_sub.todense()))
y = np.nan_to_num(y)

for i in range(X.shape[1]):
	col = X[:,i]
	pctile_upper = np.percentile(col, 99)
	pctile_lower = np.percentile(col, 1)
	X[:,i] = np.clip(col, pctile_lower, pctile_upper)

print("Scaling to unit variance...\n\n")
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_sub = scaler.transform(X_sub)

print("twosigma: Linear Reg...\n") 
clf = Ridge()
#clf = ElasticNet(alpha=0.00001, l1_ratio=0.2)

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros(X.shape[0])
sub_pred = np.zeros(X_sub.shape[0])

for i, (tr_ix, val_ix) in enumerate(kf):
	clf.fit(X[tr_ix], y[tr_ix])
	pred = clf.predict(X[val_ix])
	oob_pred[val_ix] = np.clip(pred,0,2)
	sub_pred += clf.predict(X_sub)
	print(oob_pred[val_ix][:10])
	print(y[val_ix][:10])
	scr[i] = mean_squared_error(y[val_ix], oob_pred[val_ix])
	print(np.max(oob_pred[val_ix]), np.min(oob_pred[val_ix]))
	print('Train score is:', scr[i])

sub_pred = sub_pred / 5
oob_pred_filename = '../output/oob_pred_linreg_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_linreg_' + str(np.mean(scr))
pkl.dump(oob_pred[:, None], open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred[:, None], open(sub_pred_filename + '.p', 'wb'))

