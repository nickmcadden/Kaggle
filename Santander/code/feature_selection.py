import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
import data
import argparse
from tabulate import tabulate

print("reading the train data\n")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

labels = train['TARGET']
train_ids = train['ID']
train.drop(['ID','TARGET'], axis=1, inplace=True)

# remove constant columns
for col in train.columns:
	if train[col].std() == 0:
		train.drop([col], axis=1, inplace=True)
		test.drop([col], axis=1, inplace=True)

train_summary = pd.DataFrame()
test_summary = pd.DataFrame()

for c in train.columns.values:
	print(c)
	uniques_train = len(np.unique(train[c]))
	uniques_test = len(np.unique(test[c]))
	if train[c].dtype.name in "object":
		train_meta = {"var": c, "type": train[c].dtype.name, "uniq": uniques_train, "miss": 100*(1-float(train[c].count())/float(len(train[c]))), "mean": np.nan, "min": np.min(train[c]), "max": np.max(train[c]), "stdev": np.nan}
		test_meta = {"var": c, "type": test[c].dtype.name, "uniq": uniques_test, "miss": 100*(1-float(test[c].count())/float(len(train[c]))), "mean": np.nan, "min": np.min(test[c]), "max": np.max(test[c]), "stdev": np.nan}	
	else:
		train_meta = {"var": c, "type": train[c].dtype.name, "uniq": uniques_train, "miss": 100*(1-float(train[c].count())/float(len(train[c]))), "mean": round(np.mean(train[c]),2), "min": np.min(train[c]), "max": round(np.max(train[c]),2), "stdev": np.std(train[c])}
		test_meta = {"var": c, "type": test[c].dtype.name, "uniq": uniques_test, "miss": 100*(1-float(test[c].count())/float(len(train[c]))), "mean": round(np.mean(test[c]),2), "min": np.min(test[c]), "max": np.max(test[c]), "stdev": np.std(test[c]), "stdev": np.std(test[c])}
	train_summary = train_summary.append(train_meta, ignore_index = True)
	test_summary = test_summary.append(test_meta, ignore_index = True)


print("\n\n5. Scaling to unit variance...\n")
colnames = train.columns.values.tolist()
scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(train))
train.columns = colnames

print("training a RF classifier\n")
train = train.fillna(-999)
varscores = np.zeros(train.shape[1])
for i in range(0, 30):
	clf = RandomForestClassifier(n_estimators=10, max_depth=12, oob_score=True, verbose=1, n_jobs = -1)
	clf.fit(train.values, labels)
	print(clf.oob_score_)
	varscores = varscores + clf.feature_importances_
	imp = pd.DataFrame({'var': train.columns.values, 'score': varscores}).sort('score', ascending=False)
	imp.index = range(1, len(imp) + 1)
	with pd.option_context('display.max_rows', 2000, 'display.max_columns', 3):
		print(imp)

summary = pd.merge(imp, train_summary, how='outer',on='var').sort('score',ascending = False)
summary = pd.merge(summary, test_summary, how='outer',on='var')

out_vars = ["score","var","stdev_x","stdev_y","uniq_x","uniq_y","min_x","min_y","mean_x","mean_y","max_x","max_y"]
with pd.option_context('display.max_rows', 2000, 'display.max_columns', 10):
	print(tabulate(summary[out_vars], headers="keys", tablefmt="rst"))

print("pickling importances\n")
imp.to_pickle("../input/vars_importance.pkl")
