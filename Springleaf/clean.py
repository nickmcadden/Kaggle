import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import gc
import tables
import time
import datetime as dt
import holidays
from tabulate import tabulate


print("reading the train data sample\n")
train = pd.read_hdf('input/train.h5','train')

print(train.shape)

train.drop_duplicates(inplace=True)

print(train.shape)

exit()

print("filtering by pickled important columns...\n")
vars = pd.read_pickle("input/vars_importance.pkl")
cols = list(vars.ix[0:2000,"var"])

for c in vars['var'].tolist():
	if c in ['VAR_0212']:
		train[c]= train[c].fillna(-1)
		uniques = np.unique(train[c])
		rank = vars[vars['var']==c]
		print(c, len(uniques), int(rank.index.values))
		train[c]= train[c].fillna(-1)
		freqs = train[c].value_counts().reset_index().astype(int)
		freqs.columns = [c,'freq']
		uniques = np.unique(train[c])
		corr = pd.DataFrame(pd.concat([train[c].astype(int),train['target']], axis=1).groupby(c).mean())
		summary = pd.merge(corr.reset_index(),freqs,how='outer',on=c).sort('freq',ascending = False).astype('int64')
		print(summary[c].dtype)
		print(tabulate(summary[summary['freq']>1],headers="keys",tablefmt="rst"))
exit()
