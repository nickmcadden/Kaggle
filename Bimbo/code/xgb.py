import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data as data
import argparse
import pickle as pkl
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=100)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=150)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.02)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='tgtrate')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X_train, y_train, X_sub, ids = data.load(m_params)

print("Bimbo: XGB regression...\n")
xgb_param = {'silent' : 1, 'max_depth' : 7, 'eval_metric': 'rmse', 'objective': 'reg:linear', 'eta': m_params['eta'], 'min_child_weight': 2, 'subsample': 0.7, 'colsample_bytree': 0.5}

if m_params['cv']:
	val_ix = (X_train[:,0] > 7)
	X_val = X_train[val_ix==1]
	y_val = y_train[val_ix==1]
	X_train = X_train[val_ix==0]
	y_train = y_train[val_ix==0]
	dtrain = xgb.DMatrix(X_train, np.log1p(y_train))
	dtest = xgb.DMatrix(X_sub)
	dval = xgb.DMatrix(X_val, np.log1p(y_val))
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dval,'val']))
	pred = clf.predict(dval)
	scr = np.sqrt(mean_squared_error(np.log1p(y_val), np.array(pred)))
	print('Train score is:', scr)

else:
	# Train on full data
	dtrain = xgb.DMatrix(X_train, np.log1p(y_train))
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dtrain,'train']))

	pred = clf.predict(dtrain)
	print('Train score is:', np.sqrt(mean_squared_error(np.log1p(y_train), np.array(pred))))

	model_pathname = '../output/pred_xgb_' + str(m_params['n_rounds'])
	clf.save_model(model_pathname + '.model')
	pred = np.expm1(clf.predict(dtest))

	pkl.dump(pred, open(model_pathname + '.p', 'wb'))

	print("Saving Results.")
	preds = pd.DataFrame({"id": ids, "Demanda_uni_equil": pred})
	preds.to_csv(model_pathname + '.csv', index=False)
