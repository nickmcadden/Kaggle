import data as data
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle as pkl
import argparse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1000)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=5000)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.002)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='tgtrate')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')

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

print("KNN...\n\n")
clf = KNeighborsRegressor(n_neighbors=30, weights='distance', leaf_size=20, p=1, metric='minkowski', n_jobs=-1)

kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros(X.shape[0])
sub_pred = np.zeros((X_sub.shape[0], 5))
for i, (tr_ix, val_ix) in enumerate(kf):
	clf.fit(X[tr_ix], y[tr_ix])
	pred = clf.predict(X[val_ix])
	oob_pred[val_ix] = np.array(pred)
	sub_pred[:,i] = clf.predict(X_sub)
	scr[i] = mean_squared_error(y[val_ix], np.array(pred))
	print('Train score is:', scr[i])
print(mean_squared_error(y, oob_pred))
print oob_pred[1:10]

sub_pred = sub_pred.mean(axis=1)
oob_pred_filename = '../output/oob_pred_knn_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_knn_' + str(np.mean(scr))
pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))

