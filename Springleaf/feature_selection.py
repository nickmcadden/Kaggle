import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import gc
import tables
import time
import pickle
import data_features
import argparse

parser = argparse.ArgumentParser(description='XGBoost for Springleaf')
parser.add_argument('-f','--n_features', help='Number of features', type=int, default=0)
parser.add_argument('-n','--n_rounds', help='Number of Boost iterations', type=int, default=20)
parser.add_argument('-e','--eta', help='Learning rate', type=float, default=0.1)
parser.add_argument('-r','--r_seed', help='Set random seed', type=int, default=1)
parser.add_argument('-b','--minbin', help='Minimum categorical bin size', type=int, default=1)
parser.add_argument('-cv','--cv', action='store_true')
parser.add_argument('-codetest','--codetest', action='store_true')
parser.add_argument('-getcached', '--getcached', action='store_true')
parser.add_argument('-extra', '--extra', action='store_true')

m_params = vars(parser.parse_args())

print("reading the train and test data\n")
train, target, test_sub, ids = data_features.load(m_params)
features = train.columns.values
print("training a RF classifier\n")

varscores = np.zeros(train.shape[1])
for i in range(0,10):
	clf = RandomForestClassifier(n_estimators=200, max_depth=10, oob_score=True, verbose=1, n_jobs = -1)
	clf.fit(train.values, target.values)
	print(clf.oob_score_)
	varscores = varscores + clf.feature_importances_
	imp = pd.DataFrame({'var': features, 'score': varscores}).sort('score',ascending=False)
	imp.index = range(1,len(imp) + 1)
	with pd.option_context('display.max_rows', 2000, 'display.max_columns', 3):
		print(imp)

print("pickling importances\n")
imp.to_pickle("input/vars_importance2.pkl")
