import data2 as data
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle as pkl
import argparse
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss

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

xgb_param = {'silent' : 1, 'eta': 0.02, 'max_depth':5, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'subsample': 0.7, 'num_class': 3, 'min_child_weight': 1, 'colsample_bytree': 0.7}

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros((X.shape[0], 3))
sub_pred = np.zeros((X_sub.shape[0], 3))
dtest = xgb.DMatrix(X_sub)
for i, (tr_ix, val_ix) in enumerate(kf):
	# get custom lookup for this fold
	print("CV fold: %d\n" %i)
	dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix])
	dval = xgb.DMatrix(X[val_ix], y[val_ix])
	clf = xgb.train(xgb_param, dtrain, 1700, evals=([dtrain,'train'], [dval,'val']))
	pred = clf.predict(dval)
	oob_pred[val_ix] = np.array(pred)
	sub_pred += clf.predict(dtest)
	scr[i] = log_loss(y[val_ix], np.array(pred))
	print('CV score is: %f' % scr[i])

print('Avg score is:', np.mean(scr))

sub_pred = sub_pred / 5
oob_pred_filename = '../output/oob_pred_xgb_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_xgb_' + str(np.mean(scr))
pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
preds = pd.DataFrame({"listing_id": ids, "high": sub_pred[:,0], "medium": sub_pred[:,1], "low": sub_pred[:,2]})
preds = preds[["listing_id", "high", "medium", "low"]]
preds.to_csv('../output/xgb_' + str(np.mean(scr)) + '.csv', index=False)
