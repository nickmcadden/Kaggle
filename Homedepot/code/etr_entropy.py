import data
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle as pkl
import argparse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor

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

clf = ExtraTreesRegressor(n_estimators=1000, verbose=1, max_features= 15, min_samples_split= 4, max_depth= 15, min_samples_leaf= 2, n_jobs = -1)      

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=10, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros(X.shape[0])
sub_pred = np.zeros((X_sub.shape[0], 10))

for i, (tr_ix, val_ix) in enumerate(kf):
	clf.fit(X[tr_ix], y[tr_ix])
	pred = np.clip(clf.predict(X[val_ix]),1,3)
	oob_pred[val_ix] = np.array(pred)
	sub_pred[:,i] = np.clip(clf.predict(X_sub),1,3)
	scr[i] = np.sqrt(mean_squared_error(y[val_ix], np.array(pred)))
	print('Train score is:', scr[i])
print(np.sqrt(mean_squared_error(y, oob_pred)))
print oob_pred[1:10]
sub_pred = sub_pred.mean(axis=1)
oob_pred_filename = '../output/oob_pred_etrentropy_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_etrentropy_' + str(np.mean(scr))
pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
preds = pd.DataFrame({"id": ids, "relevance": sub_pred})
preds.to_csv(sub_pred_filename + '.csv', index=False)
