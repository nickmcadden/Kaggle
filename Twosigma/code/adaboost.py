import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data2 as data
import argparse
import pickle as pkl
from scipy import stats
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import log_loss

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

print("Two Sigma: Adaboost Classifier...\n") 
clf =  AdaBoostClassifier(RandomForestClassifier(max_features=50, n_estimators=100, criterion= 'entropy', min_samples_split= 4, max_depth=21, n_jobs=-1), learning_rate=1, n_estimators=10, random_state=1)

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros((X.shape[0], 3))
sub_pred = np.zeros((X_sub.shape[0], 3))
for i, (tr_ix, val_ix) in enumerate(kf):
	clf.fit(X[tr_ix], y[tr_ix])
	pred = clf.predict_proba(X[val_ix].toarray())
	oob_pred[val_ix] = np.array(pred)
	sub_pred += clf.predict_proba(X_sub.toarray()) / 5
	scr[i] = log_loss(y[val_ix], np.array(pred))
	print('Train score is:', scr[i])
print(log_loss(y, oob_pred))

oob_pred_filename = '../output/oob_pred_adaboost_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_adaboost_' + str(np.mean(scr))
pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
preds = pd.DataFrame({"listing_id": ids, "high": sub_pred[:,0], "medium": sub_pred[:,1], "low": sub_pred[:,2]})
preds = preds[["listing_id", "high", "medium", "low"]]
preds.to_csv('../output/adaboost_' + str(np.mean(scr)) + '.csv', index=False)
