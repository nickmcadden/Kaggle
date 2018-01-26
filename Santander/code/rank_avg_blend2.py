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

sub_models = [	'pred_0.8410.csv',
				'pred_0.8429?.csv',
				'pred_med_0.8407.csv',
				'pred_0.8398.csv']

sub_cols = np.zeros((X_sub.shape[0], len(sub_models)))

for i, sub_model_name in enumerate(sub_models):
	sub_model = pd.read_csv('../output/' + sub_model_name)
	sub_cols[:,i] = sub_model.ix[:,1]

sub_cols = pd.DataFrame(sub_cols)
sub_rank = sub_cols.rank(method='min')

sub_avg = (sub_rank.ix[:,0]*0.2 + sub_rank.ix[:,1]*0.4 + sub_rank.ix[:,2]*0.2 + sub_rank.ix[:,3]*0.2) / sub_rank.shape[0]

print("Saving Results.")
model_pathname = '../output/pred_rank_avg_final3b'
preds = pd.DataFrame({"ID": ids, "TARGET": sub_avg})
preds.to_csv(model_pathname + '.csv', index=False)
