import sys
import pandas as pd
import numpy as np
import scipy as sp
import data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate

train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

labels = train['target'].values
test_ids = test['ID']
train_ids = train['ID']
train.drop(['ID','target'], axis=1, inplace=True)
test.drop(['ID'], axis=1, inplace=True)

colsToRemove = ['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']
#train.drop(colsToRemove, axis=1, inplace=True)
#test.drop(colsToRemove, axis=1, inplace=True)

print("1. Breaking dataframe into numeric, object and date parts...\n")
train_continuous = train.select_dtypes(include=['float64'])
test_continuous = test.select_dtypes(include=['float64'])

train_continuous = train.select_dtypes(include=['float64'])
test_continuous = test.select_dtypes(include=['float64'])

test_discrete = test.select_dtypes(include=['int64'])
train_discrete = train.select_dtypes(include=['int64'])

print(train_continuous.columns)

# Get only top n features
print("1b. Filtering by pickled important columns...\n")
cols = pd.read_pickle("../input/vars_importance.pkl")
topcols = cols.ix[0:131, "var"]
topcols = list(set(topcols) & set(train_continuous.columns))

print(topcols)

summary = list()

for c1 in topcols:
	train_new_eng = train[topcols]
	print(c1)
	
	for c2 in topcols:
		if c1 == c2:
			continue
		train_new_eng[c1+'*2/'+c2] = train[c1]*train[c1] / np.clip(train[c2], 0.001, 10000)

	print("training a RF classifier")
	train_new_eng = train_new_eng.fillna(-999)
	clf = RandomForestClassifier(n_estimators=60, max_depth=12, oob_score=True, verbose=0, n_jobs = -1)
	clf.fit(train_new_eng.values, labels)
	print(clf.oob_score_)
	imp = pd.DataFrame({'var': train_new_eng.columns.values, 'score': clf.feature_importances_}).sort_values(by='score', ascending=False)
	imp = imp.set_index('var')

	for c2 in topcols:
		if c1 == c2:
			continue
		if imp.ix[c1+'*2/'+c2, 'score'] / sp.maximum(imp.ix[c1, 'score'], imp.ix[c2, 'score']) > 1.6:
			row = [(c1+'*2/'+c2, imp.ix[c1+'*2/'+c2, 'score'], imp.ix[c1+'*2/'+c2, 'score'] / sp.maximum(imp.ix[c1, 'score'], imp.ix[c2, 'score']))]
			summary += row
			print(row)
	print("\n")

with pd.option_context('display.max_rows', 2000, 'display.max_columns', 10):
	summary = pd.DataFrame(summary)
	summary.columns = ['var','score','vs single']
	print(tabulate(summary, headers="keys", tablefmt="rst"))


