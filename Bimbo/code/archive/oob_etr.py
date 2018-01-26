import sys
import pandas as pd
import numpy as np
import scipy as sp
import data_oob as data
import argparse
import pickle as pkl
from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cross_validation import StratifiedShuffleSplit, LabelKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1200)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=2000)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.01)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=3)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-ct','--cat_trans', help='Category transformation method', type=str, default='std')
parser.add_argument('-cvweek','--cvweek', type=int, default=3)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)

print("Bimbo: Extra Trees...\n") 
clf =  ExtraTreesRegressor(n_estimators=500, verbose=1, max_features=7, min_samples_split=3, max_depth=12, n_jobs=-1, min_samples_leaf=2)

if m_params['cv']:
	# do cross validation scoring
	scr = np.zeros([7])
	oob_pred = np.zeros(X.shape[0])
	sub_pred = np.zeros((X_sub.shape[0], 7))
	for i in range(0,7):
		# Load data
		m_params['cvweek'] = i+3
		print('loading data for cv week',m_params['cvweek'])
		X, y, X_sub, ids = data.load(m_params)
		val_ix = (X[:,0]==m_params['cvweek'])
		tr_ix = (X[:,0]!=m_params['cvweek'])

		clf.fit(X[tr_ix], np.log1p(y[tr_ix]))
		pred = clf.predict(X[val_ix])
		oob_pred[val_ix] = np.array(pred)
		sub_pred[:,i] = clf.predict(X_sub)
		scr[i] = np.sqrt(mean_squared_error(np.log1p(y[val_ix]), np.array(pred)))
		print('Train score is:', scr[i])
	print(np.sqrt(mean_squared_error(np.log1p(y), oob_pred)))
	print oob_pred[:10]
	sub_pred = sub_pred.mean(axis=1)
	oob_pred_filename = '../output/oob_pred_etr_' + str(np.mean(scr))
	sub_pred_filename = '../output/sub_pred_etr_' + str(np.mean(scr))
	pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
	pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
	preds = pd.DataFrame({"id": ids, "Demanda_uni_equil": sub_pred})
	preds.to_csv(sub_pred_filename + '.csv', index=False)

	'''
	fraction_of_positives, mean_predicted_value = calibration_curve(y, oob_pred, n_bins=10)
	plt.plot(mean_predicted_value, fraction_of_positives, "s-",label="ssafaasf")
	plt.show()
	'''
else:
	# Train on full data
	dtrain = xgb.DMatrix(X,y)
	dtest = xgb.DMatrix(X_sub)
	clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'])

	pred = clf.predict(dtrain)
	print('Train score is:', np.sqrt(mean_squared_error(np.log1p(y), np.array(pred))))

	pred = clf.predict(dtest)

	print("Saving Results.")
	preds = pd.DataFrame({"id": ids, "Demanda_uni_equil": pred})
	preds.to_csv(model_name + '.csv', index=False)
