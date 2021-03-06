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
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=5000)
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
X, y, X_sub, ids = data.load(m_params)

print("Santander: classification...\n")
xgb_param = {'silent' : 1, 'max_depth' : 5, 'eval_metric': 'rmse', 'objective': 'reg:linear', 'eta': m_params['eta'], 'min_child_weight': 3, 'subsample': 0.7, 'colsample_bytree': 0.5}

if m_params['cv']:
	# do cross validation scoring
	kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros(X.shape[0])
	sub_pred = np.zeros((X_sub.shape[0], 5))
	dtest = xgb.DMatrix(X_sub)
	for i, (tr_ix, val_ix) in enumerate(kf):
		dtrain = xgb.DMatrix(X[tr_ix], np.log1p(y[tr_ix]))
		dval = xgb.DMatrix(X[val_ix], np.log1p(y[val_ix]))
		clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dval,'val']))
		pred = clf.predict(dval)
		oob_pred[val_ix] = np.array(pred)
		sub_pred[:,i] = clf.predict(dtest)
		scr[i] = np.sqrt(mean_squared_error(np.log1p(y[val_ix]), np.array(pred)))
		print('Train score is:', scr[i])
	print np.mean(scr)
	print sub_pred[1:10]
	sub_pred = sub_pred.mean(axis=1)
	oob_pred_filename = '../output/oob_pred_xgb_' + str(np.mean(scr))
	sub_pred_filename = '../output/sub_pred_xgb_' + str(np.mean(scr))
	pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
	pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
	preds = pd.DataFrame({"ID": ids, "target": sub_pred})
	preds.to_csv(sub_pred_filename + '.csv', index=False)

else:
	# Train on full data
	dtrain = xgb.DMatrix(X,np.log1p(y))
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dtrain,'train']))

	pred = clf.predict(dtrain)
	print('Train score is:', np.sqrt(mean_squared_error(np.log1p(y), np.array(pred))))

	model_pathname = '../output/pred_xgb_' + str(m_params['n_rounds'])
	clf.save_model(model_pathname + '.model')
	pred = clf.predict(dtest)

	pkl.dump(pred, open(model_pathname + '.p', 'wb'))

	print("Saving Results.")
	preds = pd.DataFrame({"id": ids, "Demanda_uni_equil": pred})
	preds.to_csv(model_pathname + '.csv', index=False)