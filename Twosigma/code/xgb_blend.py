import sys
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import data3 as data
import argparse
import pickle as pkl
from scipy import sparse
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.metrics import log_loss

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1200)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=400)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.1)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='freq')
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)

print("Loading OOB predictions...\n")

oob_models = [	'oob_pred_rfentropy_0.558171910933.p',
				'oob_pred_xgb_reg_0.210871555251.p',
				'oob_pred_xgb_reg_0.213364691979.p',
				'oob_pred_gbr_reg_0.228964054965.p',
				'oob_pred_etcentropy_0.551557316458.p',
				'oob_pred_adaboost_0.592650009197.p',
				#'oob_nnet_0.574446898612.p',
				#'oob_pred_linreg_0.245916676855.p',
				'oob_pred_lr_0.574865487204.p',
				'oob_pred_gbcentropy_0.550568716867.p',
				'oob_pred_gbcentropy_0.56107032581.p',
				#'oob_pred_xgb_0.522546789983.p',
				'oob_pred_xgb_0.524605390397.p',
				'oob_pred_xgb_0.524458909366.p',
				'oob_pred_xgb_0.525800613352.p',
				'oob_pred_xgb_0.529575127857.p']

X=X[:,:14]
X_sub=X_sub[:,:14]
 
for oob_model_name in oob_models:
	sub_model_name = oob_model_name.replace('oob', 'sub')
	oob_model = pkl.load(open('../output/' + oob_model_name,'rb'))
	sub_model = pkl.load(open('../output/' + sub_model_name,'rb'))
	X = np.hstack([X, oob_model])
	X_sub = np.hstack([X_sub, sub_model])

print("Two Sigma: classification...\n") 
xgb_param = {'silent' : 1, 'eta': 0.02, 'max_depth':4, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'subsample': 0.7, 'num_class': 3, 'min_child_weight': 1, 'colsample_bytree': 0.7}

print(m_params['cv'])
if m_params['cv']==True:

	# do cross validation scoring
	kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros((X.shape[0], 3))
	for i, (tr_ix, val_ix) in enumerate(kf):
		dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix])
		dval = xgb.DMatrix(X[val_ix], y[val_ix])
		clf = xgb.train(xgb_param, dtrain, 400, evals=([dtrain,'train'], [dval,'val']))
		pred = clf.predict(dval)
		oob_pred[val_ix] = np.array(pred)
		scr[i] = log_loss(y[val_ix], np.array(pred))
		print('Train score is:', scr[i])
	print oob_pred[1:10]
	print np.mean(scr)
	oob_filename = '../output/oob_blend_xgb_' + str(np.mean(scr)) + '.p'
	pkl.dump(oob_pred, open(oob_filename, 'wb'))

else:

	# Train on full data
	dtrain = xgb.DMatrix(X,y)
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, 400, evals=([dtrain,'train'], [dtrain,'train']))

	print('Predicting on test set...')
	pred = clf.predict(dtest)

	print("Saving Results.")
	model_pathname = '../output/pred_xgb_blend_' + str(m_params['n_rounds'])
	clf.save_model(model_pathname + '.model')
	preds = pd.DataFrame({"listing_id": ids, "high": pred[:,0], "medium": pred[:,1], "low": pred[:,2]})
	preds = preds[["listing_id", "high", "medium", "low"]]
	preds.to_csv(model_pathname + '.csv', index=False)
	pkl.dump(pred, open(model_pathname + '.p', 'wb'))
