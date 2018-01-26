import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import data
import argparse
from tabulate import tabulate

print("reading the train data\n")
train = pd.read_csv('../input/train.csv')

labels = train['target']
train_ids = train['ID']
train.drop(['ID','target'], axis=1, inplace=True)

print("1c. Breaking dataframe into numeric, object and date parts...\n")
train_continuous = train.select_dtypes(include=['float64'])
train_continuous['cont_max'] = train_continuous.max(axis=1)
train_continuous['cont_min'] = train_continuous.min(axis=1)
train_continuous['cont_median'] = train_continuous.median(axis=1)
train_continuous['cont_std'] = train_continuous.std(axis=1)
train_continuous['zeros'] = (train_continuous == 0).astype(int).sum(axis=1)

train_discrete = train.select_dtypes(include=['int64'])
train_discrete['dis_max'] = train_discrete.max(axis=1)
train_discrete['dis_std'] = train_discrete.std(axis=1)
train_discrete['dis_median'] = train_discrete.median(axis=1)
train_discrete['zeros'] = (train_discrete == 0).astype(int).sum(axis=1)

train_categoric = train.select_dtypes(include=['object'])

print("3. Merging arrays together...\n")
# put seperate parts together again
train = pd.concat([train_categoric, train_discrete, train_continuous], axis=1)

summary = pd.DataFrame()
for c in train.columns.values:
	print(c)
	uniques = len(np.unique(train[c]))
	if train[c].dtype.name in "object":
		meta = {"var": c, "type": train[c].dtype.name, "unique": uniques, "missing": 100*(1-float(train[c].count())/float(len(train[c]))), "median": np.nan, "min": np.min(train[c]), "max": np.max(train[c])}
	else:
		meta = {"var": c, "type": train[c].dtype.name, "unique": uniques, "missing": 100*(1-float(train[c].count())/float(len(train[c]))), "median": round(np.nanmedian(train[c]),2), "min": round(np.min(train[c]),2), "max": round(np.max(train[c]),2)}
	summary = summary.append(meta, ignore_index = True)

# Deal with categorical data
print("2. Categorical variable encoding... \n")
train_categoric = train_categoric.fillna('NA')
for c in train_categoric.columns:
	lbl = LabelEncoder()
	lbl.fit(list(train_categoric.ix[:,c]))
	train_categoric.ix[:,c] = lbl.transform(train_categoric.ix[:,c])

print("3. Merging arrays together...\n")
# put seperate parts together again
train = pd.concat([train_categoric, train_discrete, train_continuous], axis=1)

print("training a RF classifier\n")
train = train.fillna(-999)
varscores = np.zeros(train.shape[1])
for i in range(0, 10):
	clf = RandomForestClassifier(n_estimators=30, max_depth=12, oob_score=True, verbose=1, n_jobs = -1)
	clf.fit(train.values, labels)
	print(clf.oob_score_)
	varscores = varscores + clf.feature_importances_
	imp = pd.DataFrame({'var': train.columns.values, 'score': varscores}).sort('score', ascending=False)
	imp.index = range(1, len(imp) + 1)
	with pd.option_context('display.max_rows', 2000, 'display.max_columns', 3):
		print(imp)

summary = pd.merge(imp, summary, how='outer',on='var').sort('score',ascending = False)

with pd.option_context('display.max_rows', 2000, 'display.max_columns', 10):
	print(tabulate(summary, headers="keys", tablefmt="rst"))

print("pickling importances\n")
imp.to_pickle("../input/vars_importance.pkl")
