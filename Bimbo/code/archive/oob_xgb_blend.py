import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data_oob as data
import argparse
import pickle as pkl
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit, LabelKFold
from sklearn.metrics import mean_squared_error


parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=100)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=150)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.02)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='tgtrate')
parser.add_argument('-cvweek','--cvweek', type=int, default=3)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)

print("Loading OOB predictions...\n") 

oob_models = [	'oob_pred_etr_0.474376596855.p',
				'oob_pred_rfr_0.47807933038.p',
				'oob_pred_xgb_0.462796069118.p']

for oob_model_name in oob_models:
	sub_model_name = oob_model_name.replace('oob', 'sub')
	oob_model = pkl.load(open('../output/' + oob_model_name,'rb'))
	sub_model = pkl.load(open('../output/' + sub_model_name,'rb'))	
	X = np.column_stack((X, oob_model))
	X_sub = np.column_stack((X_sub, sub_model))

print("Bimbo: XGB...\n") 
xgb_param = {'silent' : 1, 'max_depth' : 10, 'eval_metric': 'rmse', 'objective': 'reg:linear', 'eta': m_params['eta'], 'min_child_weight': 2, 'subsample': 0.7, 'colsample_bytree': 0.5}

if m_params['cv']:
	# do cross validation scoring
	scr = np.zeros([7])
	oob_pred = np.zeros(X.shape[0])
	sub_pred = np.zeros((X_sub.shape[0], 7))
	dtest = xgb.DMatrix(X_sub)
	for i in range(0,7):
		# Load data
		m_params['cvweek'] = i+3
		print('loading data for cv week',m_params['cvweek'])
		X, y, X_sub, ids = data.load(m_params)
		X = np.column_stack((X, oob_model))
		val_ix = (X[:,0]==m_params['cvweek'])
		tr_ix = (X[:,0]!=m_params['cvweek'])

		dtrain = xgb.DMatrix(X[tr_ix], np.log1p(y[tr_ix]))
		dval = xgb.DMatrix(X[val_ix], np.log1p(y[val_ix]))
		clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], evals=([dtrain,'train'], [dval,'val']))
		pred = clf.predict(dval)
		oob_pred[val_ix] = np.array(pred)
		scr[i] = np.sqrt(mean_squared_error(np.log1p(y[val_ix]), np.array(pred)))			
		print('Train score is:', scr[i])
	print oob_pred[1:10]
	print np.mean(scr)

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

