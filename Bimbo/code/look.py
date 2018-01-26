import pandas as pd
import numpy as np
from tabulate import tabulate

print("reading the train data sample\n")
train = pd.read_hdf('../input/train.h5', 'train')

print(train[:100])

for c in train.columns:
	if train[c].dtype.name in "float64":
		train[c]= train[c].fillna(-909)
		uniques = np.unique(train[c])
		print(c, train[c].dtype.name, len(uniques))
	freqs = train[c].value_counts().reset_index()
	freqs.columns = [c,'freq']
	corr = pd.DataFrame(pd.concat([train[c],train['target']], axis=1).groupby(c).mean())
	summary = pd.merge(corr.reset_index(),freqs,how='outer',on=c).sort(c,ascending = True)
	print(tabulate(summary,headers="keys", tablefmt="rst"))
exit()

for c in vars['var'].tolist():
	if c in ['VAR_0483', 'VAR_0484', 'VAR_0485', 'VAR_0486', 'VAR_0487', 'VAR_0488', 'VAR_0489']:
		train[c]= train[c].fillna(-1)
		uniques = np.unique(train[c])
		rank = vars[vars['var']==c]
		print(c, len(uniques), int(rank.index.values))
		freqs = train[c].value_counts().reset_index().astype(int)
		freqs.columns = [c,'freq']
		corr = pd.DataFrame(pd.concat([train[c].astype(int),train['target']], axis=1).groupby(c).mean())
		summary = pd.merge(corr.reset_index(),freqs,how='outer',on=c).sort(c,ascending = False).astype('int64')
		print(summary[c].dtype)
		print(tabulate(summary[summary['freq']>110],headers="keys",tablefmt="rst"))
exit()

for c in vars['var'].tolist():
	if c in train.columns and train[c].dtype.name in "object":
		train[c]= train[c].fillna(-1)
		uniques = np.unique(train[c])
		if len(uniques) > 10:
			rank = vars[vars['var']==c]
			print(c, len(uniques), int(rank.index.values))
			train[c]= train[c].fillna("N/A")
			freqs = train[c].value_counts().reset_index()
			freqs.columns = [c,'freq']
			uniques = np.unique(train[c])
			corr = pd.DataFrame(pd.concat([train[c],train['target']], axis=1).groupby(c).mean())
			summary = pd.merge(corr.reset_index(),freqs,how='outer',on=c).sort('freq',ascending = False)
			print(tabulate(summary[summary['freq']>100],headers="keys",tablefmt="rst"))
exit()

for c in vars['var'].tolist():
	if c in train.columns and train[c].dtype.name == "object":
		train[c]= train[c].fillna("N/A")
		freqs = train[c].value_counts().reset_index()
		freqs.columns = [c,'freq']
		uniques = np.unique(train[c])
		corr = pd.DataFrame(pd.concat([train[c],train['target']], axis=1).groupby(c).mean())
		rank = vars[vars['var']==c]
		print(c, len(uniques), int(rank.index.values))
		summary = pd.merge(corr.reset_index(),freqs,how='outer',on=c).sort('freq',ascending = False)
		print(tabulate(summary[summary['freq']>100],headers="keys",tablefmt="rst"))
exit()

with pd.option_context('display.max_rows', 1000, 'display.max_columns', 3):
	print(vars)
		
for c in vars['var'].tolist():
	if c in train.columns and train[c].dtype.name != "object":
		uniques = np.unique(train[c])
		if len(uniques) < 300:
			freqs = train[c].value_counts().reset_index()
			freqs.columns = [c,'freq']
			corr = pd.DataFrame(pd.concat([train[c],train['target']], axis=1).groupby(c).mean())
			rank = vars[vars['var']==c]
			print(c, len(uniques), int(rank.index.values))
			summary = pd.merge(corr.reset_index(),freqs,how='outer',on=c).sort('freq',ascending = False)
			print(tabulate(summary[summary['freq']>30],headers="keys",tablefmt="rst"))

