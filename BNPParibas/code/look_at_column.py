import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from time import time
from tabulate import tabulate
from sklearn.cross_validation import StratifiedKFold

def prob_weight_lookup(code, lookup, labels):
	try:
		scr = np.random.normal(loc=lookup.ix[code, 'tgt_rate_adj'], scale=lookup.ix[code, 'ci'])
	except:
		scr = np.mean(labels)
	return np.clip(scr,0,1)

def category_to_prob_weight(traincol, testcol, labels):
	traincol, testcol, labels = pd.Series(traincol), pd.Series(testcol), pd.Series(labels)
	kf = StratifiedKFold(labels, n_folds=4, shuffle=True, random_state=1)
	traincolprob = np.zeros(traincol.shape[0])
	testcolprob = np.zeros(testcol.shape[0])
	for kfold, (tr_ix, val_ix) in enumerate(kf):
		train_tr = traincol.iloc[tr_ix]
		train_val = traincol.iloc[val_ix]
		freqs = pd.DataFrame(train_tr.value_counts())
		corr = pd.concat([train_tr, labels.iloc[tr_ix]], axis=1)
		corr = pd.DataFrame(corr.groupby(0).mean())
		lookup = pd.merge(corr, freqs, how='outer', left_index=True, right_index=True)
		lookup.columns = (['target','freq'])
		# Bayesian aspect - tend towards mean target % for levels with low freq count
		lookup['tgt_rate_adj'] = ((lookup['freq'] - 1) * lookup['target'] + np.mean(labels)) / lookup['freq']
		# Calc confidence interval (missing z score multiplier so just 1 sd)
		lookup['ci'] = 0.0001 * np.sqrt((lookup['tgt_rate_adj'] * (1-lookup['tgt_rate_adj']) / lookup['freq']))
		traincolprob[val_ix] = traincol.iloc[val_ix].apply(lambda row: prob_weight_lookup(row, lookup, labels.iloc[tr_ix]))
		testcolprob[:] = testcol.apply(lambda row: prob_weight_lookup(row, lookup, labels))
	return traincolprob, testcolprob

def prob_weight_lookup2(code, lookup, labels, target=np.nan):
	try:
		if target == 0:
			scr = lookup.ix[code, 'tgt0']
		elif target == 1:
			scr = lookup.ix[code, 'tgt1']
		else:
			scr = lookup.ix[code, 'target']
		if scr == 0:
			scr = np.mean(labels)
	except:
		scr = np.mean(labels)
	return np.clip(scr,0,1)

def category_to_prob_weight2(train, test, col, labels):
	print(col)
	traincol, testcol, labels = pd.Series(train[col]), pd.Series(test[col]), pd.Series(labels)
	trn = pd.concat([traincol, labels], axis=1)
	trn.columns = ([col,'target'])
	freqs = pd.DataFrame(traincol.value_counts())
	corr = pd.DataFrame(trn.groupby(col).mean())
	lookup = pd.merge(corr, freqs, how='outer', left_index=True, right_index=True)
	lookup.columns = (['target','freq'])
	lookup['tgt1'] = np.clip(((lookup['target'] * lookup['freq']) - 1) / (lookup['freq'] - 1), 0, 1)
	lookup['tgt1'] = ((lookup['freq'] - 1) * lookup['tgt1'] + np.mean(labels)) / lookup['freq']
	lookup['tgt0'] = np.clip(((lookup['target'] * lookup['freq'])) / (lookup['freq'] - 1), 0, 1)
	lookup['tgt0'] = ((lookup['freq'] - 1) * lookup['tgt0'] + np.mean(labels)) / lookup['freq']
	lookup.ix[lookup['tgt1'].isnull(), 'tgt1'] = labels.mean()
	lookup.ix[lookup['tgt0'].isnull(), 'tgt0'] = labels.mean()
	print(lookup)
	traincolprob = trn.apply(lambda row: prob_weight_lookup2(row[col], lookup, labels, row['target']), axis=1)
	testcolprob = testcol.apply(lambda row: prob_weight_lookup2(row, lookup, labels))

	return traincolprob, testcolprob

t1 = time()

train = ([1,2,3,4,4,4,4,5,5,6,6,4,3,9,3,3,2])
test = ([1,2,3,4,4,4,4,5,5,6,6,7,6,7,9,2,2])
labels = ([0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,0])
print(train)
print(test)
print(labels)

trainzz, testzz = category_to_prob_weight(train, test, labels)
print(trainzz)
print(testzz)
t2 = time()
print(t2-t1)

train = pd.DataFrame(train)
test = pd.DataFrame(test)
train.columns = ['A']
test.columns = ['A']
trainzz, testzz = category_to_prob_weight2(train, test, 'A', labels)
print(trainzz)
print(testzz)

t3 = time()
print(t3-t2)
exit()

print("reading the train data sample\n")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
labels = train['target']
col = 'v22'
#df['newcol'] = train[cols[0]] - train[cols[1]]
#print(tabulate(pd.concat([df,labels])[0:999], headers="keys", tablefmt="rst"))

train[col] = train[col].fillna('NA')
test[col] = test[col].fillna('NA')
print(test[col].head)
train[col], test[col] = category_to_prob_weight(train[col], test[col], labels)
print(test[col].head)
exit()

print(tabulate(summary[summary['freq']>=1],headers="keys",tablefmt="rst"))
print(sum(summary['freq']))
print(np.mean(labels))
summary = summary[summary['freq']>=10]
print(sum(summary['freq']))
exit()
