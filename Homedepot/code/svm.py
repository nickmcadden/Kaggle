import sys
import pandas as pd
import numpy as np
import scipy as sp
import data
import argparse
import pickle as pkl
from sklearn.metrics import mean_squared_error
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn import svm

def log_loss(act, pred):
    """ Vectorised computation of logloss """
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1, act)*sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0/len(act)
    return ll

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

print("Scaling to unit variance...\n\n")
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_sub = scaler.transform(X_sub)

print("BNP Parabas: L1 classification...\n") 
clf = svm.SVR(C=0.5)

if m_params['cv']:
	# do cross validation scoring
	kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros(X.shape[0])
	sub_pred = np.zeros((X_sub.shape[0], 5 ))
	''' Val and train sets swapped to make train the smaller one '''
	for i, (val_ix, tr_ix) in enumerate(kf):
		clf.fit(X[tr_ix], y[tr_ix])
		pred = np.clip(clf.predict(X[val_ix]),1,3)
		oob_pred[val_ix] = np.array(pred)
		sub_pred[:,i] = np.clip(clf.predict(X_sub),1,3)
		scr[i] = np.sqrt(mean_squared_error(y[val_ix], np.array(pred)))
		print('Train score is:', scr[i])
	print(np.sqrt(mean_squared_error(y, oob_pred)))
	print oob_pred[1:10]
	sub_pred = sub_pred.mean(axis=1)
	oob_pred_filename = '../output/oob_pred_svr_' + str(np.mean(scr))
	sub_pred_filename = '../output/sub_pred_svr_' + str(np.mean(scr))
	pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
	pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
	preds = pd.DataFrame({"id": ids, "relevance": sub_pred})
	preds.to_csv(sub_pred_filename + '.csv', index=False)

else:
	# Train on full data
	pred = clf.predict(X_sub)
	print('Train score is:', log_loss(y, np.array(pred))) 

	print("Saving Results.")
	preds = pd.DataFrame({"id": ids, "relevance": pred})
	preds.to_csv(model_name + '.csv', index=False)
