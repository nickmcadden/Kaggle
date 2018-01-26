import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data_new as data
import argparse
import pickle as pkl
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=300)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=500)
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
X, y, X_sub, ids, zeromap = data.load(m_params)

print("Santander: classification...\n")
xgb_param = {'silent' : 1, 'max_depth' : 5, 'eval_metric': 'auc', 'objective': 'reg:logistic', 'eta': m_params['eta'], 'min_child_weight': 3, 'subsample': 0.7, 'colsample_bytree': 0.5}

if m_params['cv']:
	dtest = xgb.DMatrix(X_sub)
	oob_pred = np.zeros((X.shape[0], 10))
	sub_pred = np.zeros((X_sub.shape[0], 50))
	scr = np.zeros((5, 10))
	for j in range(10):
		kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=j)
		cols = 90 + (j*10)
		# do cross validation scoring
		for i, (tr_ix, val_ix) in enumerate(kf):
			dtrain = xgb.DMatrix(X[tr_ix,0:cols], y[tr_ix])
			dval = xgb.DMatrix(X[val_ix,0:cols], y[val_ix])
			clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dval,'val']))
			pred = clf.predict(dval)
			oob_pred[val_ix, j] = np.array(pred)
			sub_pred[:, j*5+i] = np.array(clf.predict(dtest))
			scr[i,j] = roc_auc_score(y[val_ix], np.array(pred))
			print('Train score ist:', scr[i,j])
		print ('CV', i, 'score ',np.mean(scr, axis=0))
		#print oob_pred[1:10]
		print sub_pred[1:10]
	print("Overall mean scores", np.mean(scr, axis=0))
	sub_pred = np.median(sub_pred, axis=1) * zeromap
	oob_pred = np.median(oob_pred, axis=1)
	oob_pred_filename = '../output/oob_pred_xgb_med' + str(m_params['n_features']) + '_' + str(np.mean(scr))
	sub_pred_filename = '../output/sub_pred_xgb_med' + str(m_params['n_features']) + '_' + str(np.mean(scr))
	pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
	pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
	preds = pd.DataFrame({"ID": ids, "target": sub_pred})
	preds.to_csv(sub_pred_filename + '.csv', index=False)

else:

	for j in range(150,200):
		dtrain = xgb.DMatrix(X[:,0:j],y)
		dtest = xgb.DMatrix(X_sub[:,0:j])
		xgb_param["seed"] = j
		clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dtrain,'train']))
		pred = clf.predict(dtrain)
		print('Train score is:', roc_auc_score(y, np.array(pred)))
		pred = clf.predict(dtest) * zeromap
		print("Saving Results.")
		preds = pd.DataFrame({"ID": ids, "target": pred})
		preds.to_csv('XGB_' + str(j) + '_.csv', index=False)


