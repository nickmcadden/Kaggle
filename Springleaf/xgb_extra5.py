# Try these thiing 
#############################
# check categorical encoding to make sure the correlation type is working correctly
# cap outliers for the numerical columns
# further clean the string based features such as empty string replacement
# tune the XGboost more to accept larger dimensional input
# look at calibration curve of final output and create own adjuster
# Flag up holidays to add to the model.
# bin data to increase linearity
# Do somthing with the professions!

import sys, os
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import data5 as data
import argparse
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(description='XGBoost for Springleaf')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=1200)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=50)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.1)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=1)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')
m_params = vars(parser.parse_args())

# Load data
X, y, X_sub, ids = data.load(m_params)
X, y = shuffle(X, y, random_state = m_params['r_seed'])

print("Springleaf: binary classification...\n") 
model_name = 'output/' + "_".join(['xgb',str(m_params['n_rounds']),str(m_params['n_features']),str(m_params['eta'])])

print('Looking for ' + model_name + '.model') 
if os.path.isfile(model_name + '.model'):
	m_params['n_rounds'] = 	15000
else:
	print("model not available")
	exit()

xgb_param = {'silent' : 0, 'max_depth' : 9, 'alpha' : 4, 'eta': m_params['eta'], 'objective':'binary:logistic', 'eval_metric':'auc', 'min_child_weight':6, 'colsample_bytree':0.5, 'subsample':0.7}

# Train on full data
dtrain = xgb.DMatrix(X,y)
dtest = xgb.DMatrix(X_sub)
clf = xgb.train(xgb_param, dtrain, m_params['n_rounds'], xgb_model = model_name + '.model')

model_name = 'output/' + "_".join(['xgb',str(m_params['n_rounds']),str(m_params['n_features']),str(m_params['eta'])])
clf.save_model(model_name + '.model')
pred = clf.predict(dtest)

print("Saving Results.")
preds = pd.DataFrame({"ID": ids, "target": pred})
preds.to_csv(model_name + '.csv', index=False)
