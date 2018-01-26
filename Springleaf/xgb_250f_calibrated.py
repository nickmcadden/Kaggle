# Try these thiing 
#############################################
#############################################
# check categorical encoding to make sure the correlation type is working correctly
# cap outliers for the numerical columns
# further clean the string based features such as empty string replacement
# manually engineer extra features such as parts of zip codes and phone numbers
# tune the XGboost more to accept larger dimensional input
# look at calibration curve of final output and create own adjuster

import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import numpy as np
import gc
import tables
import time
import datetime as dt
import pickle

def cap_outliers(df):
	for i in df.columns.values.tolist():
		uniques = len(np.unique(train[i]))
		if uniques > 200:
			print(i)
			sd = np.std(df[i])
			mn = np.mean(df[i])
			df[i] = 0.5 +  (((df[i]-mn)/sd)*0.19)
			df[i] = np.clip(df[i],0,1)	
	return df

t0 = time.time()
print("reading the train and test data...\n")
train = pd.read_hdf('train_stg1.h5', 'train')
test  = pd.read_hdf('test_stg1.h5','test')

print("filtering by pickled important columns...\n")
vars = pd.read_pickle("vars_importance.pkl")
train = train[vars.ix[0:250,"var"].tolist()+['target']]
test = test[vars.ix[0:250,"var"].tolist()]

#train = cap_outliers(train)

print("Springleaf: binary classification...\n")
for i in [1,2,3]:
	train = shuffle(train, random_state=1)
	print("converting to numpy array...\n")
	X = train.values[:,:-1]
	y = train.target.values

	test_index = range(0,25000)
	train_index = range(25000,y.shape[0])

	clf = xgb.XGBClassifier(nthread=-1, objective="binary:logistic", max_depth=9, min_child_weight=6,  subsample=0.7, colsample_bytree=0.5, n_estimators=50, learning_rate=0.05)
	clf.fit(X[train_index],y[train_index], eval_set=[(X[test_index],y[test_index])], eval_metric='auc')
	pred = clf.predict_proba(X[test_index])
	print(roc_auc_score(y[test_index], pred[:,1]))

'''
clf.fit(X,y)
submission = clf.predict_proba(test.values)[:,1]
print(submission.shape, len(submission))	
print("Saving Results.")
preds = pd.DataFrame({"ID": np.array(range(len(submission)))+1, "target": submission})
preds.to_csv('xgb' + '.csv', index=False)
'''