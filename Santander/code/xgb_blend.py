import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data
import argparse
import pickle as pkl
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=120)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=1000)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='freq')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids, zeromap = data.load(m_params)

print("Loading OOB predictions...\n") 

oob_models = [	'oob_pred_rfentropy_0.837334846872.p',
				'oob_pred_xgb_0.842342706639.p',
				'oob_nnet_0.829875226933.p']

for oob_model_name in oob_models:
	sub_model_name = oob_model_name.replace('oob', 'sub')
	oob_model = pkl.load(open('../output/' + oob_model_name,'rb'))
	sub_model = pkl.load(open('../output/' + sub_model_name,'rb'))	
	X = np.column_stack((X, oob_model))
	X_sub = np.column_stack((X_sub, sub_model))


print("BNP Parabas: classification...\n") 
xgb_param = {'silent' : 1, 'max_depth' : 5, 'eval_metric' : 'auc', 'eta': m_params['eta'], 'min_child_weight': 3, 'objective': 'binary:logistic', 'subsample': 0.7, 'colsample_bytree': 0.5}

if m_params['cv']:
	# do cross validation scoring
	kf = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros(X.shape[0])

	for i, (tr_ix, val_ix) in enumerate(kf):
		dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix])
		dval = xgb.DMatrix(X[val_ix], y[val_ix])
		clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dval,'val']))
		pred = clf.predict(dval)
		oob_pred[val_ix] = np.array(pred)
		scr[i] = roc_auc_score(y[val_ix], np.array(pred))
		print('Train score is:', scr[i])
	print oob_pred[1:10]
	print np.mean(scr)
	oob_filename = '../output/oob_blend_ens' + str(np.mean(scr)) + '.p'
	pkl.dump(oob_pred, open(oob_filename, 'wb'))

else:

	# Train on full data
	dtrain = xgb.DMatrix(X,y)
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dtrain,'train']))

	print('Predicting on test set...')
	pred = clf.predict(dtest)

	print("Saving Results.")
	model_pathname = '../output/pred_xgb_blend_' + str(m_params['n_rounds'])
	clf.save_model(model_pathname + '.model')
	preds = pd.DataFrame({"ID": ids, "PredictedProb": pred})
	preds.to_csv(model_pathname + '.csv', index=False)
	pkl.dump(pred, open(model_pathname + '.p', 'wb'))
