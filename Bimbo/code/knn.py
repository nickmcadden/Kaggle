import sys
import pandas as pd
import numpy as np
import data_log_means as data
import argparse
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=90)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=2000)
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

print("Bimbo: KNN...\n") 
clf = KNeighborsRegressor(n_neighbors=200, weights='distance', algorithm='auto', leaf_size=30, n_jobs=-1) 

val_ix = (X[:,0] > 7)
tr_ix = (X[:,0] <= 7)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_sub = scaler.transform(X_sub)

if m_params['cv']:
	# do cross validation scoring
	clf.fit(X[tr_ix], np.log1p(y[tr_ix]))
	pred = clf.predict(X[val_ix])
	scr = np.sqrt(mean_squared_error(np.log1p(y[val_ix]), pred))
	print('Train score is:', scr)

else:
	# Train on full data
	clf.fit(X, np.log1p(y))
	pred = np.expm1(clf.predict(X_sub))
	print("Saving Results.")
	preds = pd.DataFrame({"id": ids, "Demanda_uni_equil": pred})
	preds.to_csv('../output/knn_results.csv', index=False)
