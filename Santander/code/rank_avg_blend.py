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
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='XGBoost for BNP')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1)
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
				'oob_pred_xgb_0.842810879746.p',
				'oob_nnet_0.829875226933.p']

model_weights = [0.2, 0.7, 0.1]

oob_cols = np.zeros((X.shape[0], len(oob_models)))
sub_cols = np.zeros((X_sub.shape[0], len(oob_models)))

for i, oob_model_name in enumerate(oob_models):
	sub_model_name = oob_model_name.replace('oob', 'sub')
	oob_model = pkl.load(open('../output/' + oob_model_name,'rb'))
	sub_model = pkl.load(open('../output/' + sub_model_name,'rb'))	
	oob_cols[:,i] = oob_model
	sub_cols[:,i] = sub_model

oob_cols = pd.DataFrame(oob_cols)
oob_rank = oob_cols.rank(method='min')
oob_rank.columns = ["rf","xgb","nnet"]

sub_cols = pd.DataFrame(sub_cols)
sub_rank = sub_cols.rank(method='min')
sub_rank.columns = ["rf","xgb","nnet"]

print(oob_cols)
print(oob_rank)

oob_avg = (oob_rank.ix[:,"rf"]*0.2 + oob_rank.ix[:,"nnet"]*0.1 + oob_rank.ix[:,"xgb"]*0.7)/oob_rank.shape[0]
sub_avg = (sub_rank.ix[:,"rf"]*0.2 + sub_rank.ix[:,"nnet"]*0.1 + sub_rank.ix[:,"xgb"]*0.7)/sub_rank.shape[0]

print roc_auc_score(y, oob_avg)
print(oob_avg)

print("Saving Results.")
model_pathname = '../output/pred_rank_avg_' + str(roc_auc_score(y, oob_avg))
preds = pd.DataFrame({"ID": ids, "TARGET": sub_avg})
preds.to_csv(model_pathname + '.csv', index=False)
