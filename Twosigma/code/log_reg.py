import sys
import pandas as pd
import numpy as np
import scipy as sp
import data_linear as data
import argparse
import pickle as pkl
from sklearn.metrics import mean_squared_error
from scipy import stats
from collections import OrderedDict
from sklearn.metrics import log_loss
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

print("Scaling to unit variance...\n\n")
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_sub = scaler.transform(X_sub)

print("twosigma: LogReg classification...\n") 
clf = LogisticRegression(C=5.0)
#clf = ElasticNet(alpha=0.001, l1_ratio=1)

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros((X.shape[0], 3))
sub_pred = np.zeros((X_sub.shape[0], 3))

for i, (tr_ix, val_ix) in enumerate(kf):
	for j in range(3):
		y_tmp = (y[tr_ix] == j).astype(int)
		clf.fit(X[tr_ix], y_tmp)
		pred = clf.predict_proba(X[val_ix])[:,1]
		oob_pred[val_ix, j] = np.clip(pred,0.01,0.99)
		sub_pred[:, j] += clf.predict_proba(X_sub)[:,1]
	rowsums = np.sum(oob_pred[val_ix], axis=1)
	oob_pred[val_ix] = oob_pred[val_ix] / rowsums.reshape((rowsums.size, 1))
	print(oob_pred[val_ix])
	scr[i] = log_loss(y[val_ix], oob_pred[val_ix])
	print('Train score is:', scr[i])

rowsums = np.sum(sub_pred, axis=1)
sub_pred = sub_pred / rowsums.reshape((rowsums.size, 1))
oob_pred_filename = '../output/oob_pred_lr_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_lr_' + str(np.mean(scr))
pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
preds = pd.DataFrame({"listing_id": ids, "high": sub_pred[:,0], "medium": sub_pred[:,1], "low": sub_pred[:,2]})
preds = preds[["listing_id", "high", "medium", "low"]]
preds.to_csv('../output/lr_' + str(np.mean(scr)) + '.csv', index=False)
