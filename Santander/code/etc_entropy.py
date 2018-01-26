import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import data
import argparse
import pickle as pkl
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='XGBoost for Santander')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=120)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=2000)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
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

print("BNP Parabas: classification...\n") 
clf = ExtraTreesClassifier(n_estimators=500, verbose=1, max_features=30, criterion= 'entropy', min_samples_split= 6, max_depth=20, n_jobs=-1, min_samples_leaf=3)

if m_params['cv']:
	kf = KFold(X.shape[0], n_folds=4, shuffle=True, random_state=1)
	scr = np.zeros([len(kf)])
	oob_pred = np.zeros(X.shape[0])
	sub_pred = np.zeros((X_sub.shape[0], 4))
	for i, (tr_ix, val_ix) in enumerate(kf):
		clf.fit(X[tr_ix], y[tr_ix])
		pred = clf.predict_proba(X[val_ix])
		oob_pred[val_ix] = np.array(pred[:,1])
		sub_pred[:,i] = clf.predict_proba(X_sub)[:,1]
		scr[i] = roc_auc_score(y[val_ix], np.array(pred[:,1]))
		print('Train score is:', scr[i])
	print(roc_auc_score(y, oob_pred))
	print oob_pred[1:10]
	sub_pred = sub_pred.mean(axis=1)
	oob_pred_filename = '../output/oob_pred_etcentropy_' + str(np.mean(scr))
	sub_pred_filename = '../output/sub_pred_etcentropy_' + str(np.mean(scr))
	pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
	pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
	preds = pd.DataFrame({"ID": ids, "PredictedProb": sub_pred})
	preds.to_csv(sub_pred_filename + '.csv', index=False)

else:
	# Train on full data
	dtrain = xgb.DMatrix(X,y)
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'])

	pred = clf.predict(dtrain)
	print('Train score is:', log_loss(y, np.array(pred))) 

	#clf.save_model(model_name + '.model')
	pred = clf.predict(dtest)

	print("Saving Results.")
	preds = pd.DataFrame({"ID": ids, "target": pred})
	preds.to_csv(model_name + '.csv', index=False)
