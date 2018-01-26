import sys
import pandas as pd
import numpy as np
import scipy as sp
import data_means as data
import argparse
import pickle as pkl
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.metrics import mean_squared_error

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

val_ix = (X[:,0] > 7)
tr_ix = (X[:,0] <= 7)

X = X[:,7:]
X_sub = X_sub[:,7:]
print(X[:10])

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_sub = scaler.transform(X_sub)

print("Bimbo: L1 classification...\n") 
clf = SGDRegressor(loss='squared_loss', penalty='elasticnet', alpha=0.1, l1_ratio=0.7, fit_intercept=True, n_iter=5)

if m_params['cv']:
	# do cross validation scoring
	clf.fit(X[tr_ix], np.log1p(y[tr_ix]))
	pred = clf.predict(X[val_ix])
	scr = np.sqrt(mean_squared_error(np.log1p(y[val_ix]), pred))
	print('Train score is:', scr)

else:
	# Train on full data
	clf.fit(X,y)
	pred = clf.predict(X_sub)
	print('Train score is:', np.sqrt(mean_squared_error(np.log1p(y[val_ix]), pred))) 

	print("Saving Results.")
	preds = pd.DataFrame({"ID": ids, "target": pred})
	preds.to_csv(model_name + '.csv', index=False)
