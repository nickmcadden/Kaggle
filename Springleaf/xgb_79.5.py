import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
import gc
import tables
import time
import datetime as dt

t0 = time.time()
print("reading the train and test data...\n")
train = pd.read_hdf('train.h5', 'train')
test  = pd.read_hdf('test.h5','test')

print('doing da date cols')

def tdtoint(td):
	if not pd.isnull(td):
		return td.astype('timedelta64[D]').astype(int)
	else:
		return 0

for i in ['VAR_0073','VAR_0075','VAR_0176','VAR_0179','VAR_0217','VAR_0169','VAR_0178','VAR_0166']:
	for j in ['VAR_0073','VAR_0075','VAR_0176','VAR_0179','VAR_0217','VAR_0169','VAR_0178','VAR_0166']:
		if i < j:
			keypair = i+'_'+j
		else:
			keypair = j+'_'+i
		if i!=j and keypair not in train.columns:
			train[keypair] = train[i] - train[j]
			train[keypair] = train[keypair].apply(tdtoint)
			test[keypair] = test[i] - test[j]
			test[keypair] = test[keypair].apply(tdtoint)

datecols = pd.read_pickle('datecols.pkl')
for c in datecols['col'].values.tolist():
	train[c+'_y']=train[c].dt.year
	train[c+'_m']=train[c].dt.month
	train[c+'_d']=train[c].dt.day
	train[c+'_wd']=train[c].dt.weekday
	train[c+'_hr']=train[c].dt.hour
	test[c+'_y']=test[c].dt.year
	test[c+'_m']=test[c].dt.month
	test[c+'_d']=test[c].dt.day
	test[c+'_wd']=test[c].dt.weekday
	test[c+'_hr']=test[c].dt.hour
train.drop(datecols['col'].values.tolist(), axis=1, inplace=True)

print("categorical variable encoding and cleaning...\n")
for c in train.columns[1:-1]:
	if train[c].dtype.name == 'object':
		freqs = train[c].append(test[c]).value_counts()
		train[c] = pd.match(train[c].values, freqs[0:70].index)
		test[c] = pd.match(test[c].values, freqs[0:70].index)
		
train = train.fillna(0)
test = test.fillna(0)

labels = train['target']
train.drop(['ID','target'], axis=1, inplace=True)
features = train.columns.values

'''
print("training a RF classifier to get best vars\n")
clf = RandomForestClassifier(n_jobs=-1,n_estimators=500, max_depth=10, oob_score=True, verbose=1)

varscores = np.zeros(train.shape[1])
for i in range(0,1):
	clf.fit(train.values, labels.values)
	print(clf.oob_score_)
	varscores = varscores + clf.feature_importances_
	imp = pd.DataFrame({'var': features, 'score': varscores}).sort('score',ascending=False)
	imp.index = range(1,len(imp) + 1)
	with pd.option_context('display.max_rows', 2000, 'display.max_columns', 3):
		print(imp)

print("pickling importances\n")
imp.to_pickle("vars_importance.pkl")
print(train.columns.values)
'''

print("filtering by pickled important columns...\n")
vars = pd.read_pickle("vars_importance.pkl")
train = train[vars.ix[0:1250,"var"].tolist()]
test = test[vars.ix[0:1250,"var"].tolist()]

print("converting to numpy array...\n")
X = train.values
y = labels.values

gc.collect()

print("Springleaf: binary classification...\n")

clf = xgb.XGBClassifier(nthread=-1, objective="binary:logistic", max_depth=9, min_child_weight=6,  subsample=0.7, colsample_bytree=0.5, n_estimators=20000, learning_rate=0.00062)
'''
kf = StratifiedShuffleSplit(y, n_iter=4, test_size=0.25, random_state=1)
scr = np.zeros([len(kf)])
for i, (train_index, test_index) in enumerate(kf):
	clf.fit(X[train_index],y[train_index], eval_set=[(X[test_index],y[test_index])],eval_metric='auc')
	pred = clf.predict_proba(X[test_index])
	scr[i] = roc_auc_score(y[test_index], pred[:,1])
	print(scr[i])
print np.mean(scr)

'''
clf.fit(X,y)
submission = clf.predict_proba(test.values)[:,1]
print(submission.shape, len(submission))	
print("Saving Results.")
preds = pd.DataFrame({"ID": np.array(range(len(submission)))+1, "target": submission})
preds.to_csv('xgb' + '.csv', index=False)
