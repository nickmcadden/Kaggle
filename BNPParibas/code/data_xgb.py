import sys
import pandas as pd
import numpy as np
import data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import StratifiedKFold

def prob_weight_lookup(code, lookup, labels):
	try:
		scr = np.random.normal(loc=lookup.ix[code, 'tgt_rate_adj'], scale=lookup.ix[code, 'ci'])
	except:
		scr = np.mean(labels)
	return np.clip(scr,0,1)

def category_to_prob_weight(train, test, col, labels):
	
	traincol, testcol, labels = pd.Series(train[col]), pd.Series(test[col]), pd.Series(labels)
	kf = StratifiedKFold(labels, n_folds=4, shuffle=True, random_state=1)
	traincolprob = np.zeros(traincol.shape[0])
	testcolprob = np.zeros(testcol.shape[0])
	print(col)
	for kfold, (tr_ix, val_ix) in enumerate(kf):
		print(kfold)
		train_tr = traincol.iloc[tr_ix]
		train_val = traincol.iloc[val_ix]
		freqs = pd.DataFrame(train_tr.value_counts())
		corr = pd.concat([train_tr, labels.iloc[tr_ix]], axis=1)
		corr = pd.DataFrame(corr.groupby(col).mean())
		lookup = pd.merge(corr, freqs, how='outer', left_index=True, right_index=True)
		lookup.columns = (['target','freq'])
		# Bayesian aspect - tend towards mean target % for levels with low freq count
		lookup['tgt_rate_adj'] = ((lookup['freq'] - 1) * lookup['target'] + np.mean(labels)) / lookup['freq']
		# Calc confidence interval (missing z score multiplier so just 1 sd)
		lookup['ci'] = 0.001 * np.sqrt((lookup['tgt_rate_adj'] * (1-lookup['tgt_rate_adj']) / lookup['freq']))
		traincolprob[val_ix] = train_val.apply(lambda row: prob_weight_lookup(row, lookup, labels))
		testcolprob[:] = testcol.apply(lambda row: prob_weight_lookup(row, lookup, labels))

	return traincolprob, testcolprob

def category_transformation(train_categoric, test_categoric, labels, type='std'):
	
	if type == 'freq':
		print("Encoding categories by freqency rank...")
		for c in train_categoric.columns:
			freqs = train_categoric[c].append(test_categoric[c]).value_counts()
			train_categoric[c] = pd.match(train_categoric[c].values, freqs[0:91].index)
			test_categoric[c] = pd.match(test_categoric[c].values, freqs[0:91].index)

	if type == 'std':
		print("Encoding categories by sklearn label encoder...")
		for c in train_categoric.columns:
			lbl = LabelEncoder()
			lbl.fit(list(train_categoric.ix[:,c]) + list(test_categoric.ix[:,c]))
			train_categoric.ix[:,c] = lbl.transform(train_categoric.ix[:,c])
			test_categoric.ix[:,c] = lbl.transform(test_categoric.ix[:,c])

	if type == 'tgtrate':
		print("Encoding categories by target rate...")
		for c in train_categoric.columns:
			train_categoric[c], test_categoric[c] = category_to_prob_weight(train_categoric, test_categoric, c, labels)

	if type == 'rank':
		print("Encoding categories by rank transformation...")
		for c in train_categoric.columns:
			rank = pd.concat([train_categoric[c],labels], axis=1).groupby(c).mean().sort_values(by='target', ascending=False)
			train_categoric[c] = pd.match(train_categoric[c].values, rank[0:20000].index)
			test_categoric[c] = pd.match(test_categoric[c].values, rank[0:20000].index)

	if type == 'onehot':
		print("One hot... ")
		for c in train_categoric.columns:
			uniques = np.unique(train_categoric[c])
			if len(uniques) > 100:
				train_categoric.drop(c, axis=1, inplace=True)
				test_categoric.drop(c, axis=1, inplace=True)
		x_cat_train = train_categoric.T.to_dict().values()
		x_cat_test = test_categoric.T.to_dict().values()

		# vectorize
		vectorizer = DV(sparse = False)
		train_categoric = pd.DataFrame(vectorizer.fit_transform(x_cat_train))
		test_categoric = pd.DataFrame(vectorizer.transform(x_cat_test))

	return train_categoric, test_categoric


def bin(train_col, test_col, target, minbin=10):
	train_col = train_col.values
	test_col = test_col.values

	long_array = np.concatenate([train_col,test_col], axis=0)
	items, freqs = np.unique(long_array, return_counts = True)

	if len(items)==1:
		return train_col, test_col

	newitems = np.copy(items)
	binitemcount, ix_start = 0, 0

	# Loop through column items and group any consecutive values occuring less frequently than minimum bin size
	for i in range(0, len(items)):
		binitemcount += freqs[i]
		if binitemcount >= minbin:
			newitems[ix_start:i+1] = ix_start
			binitemcount = 0
			ix_start = i+1
		else:
			newitems[i] = i

	if binitemcount > 0:
		newitems[ix_start:i+1] = len(items) - 1

	#print(pd.DataFrame({'freqs': freqs, 'olditem': items, 'newitems': newitems}))
	#print(tabulate(summary[summary[col.name]>=max-4], headers="keys", tablefmt="rst"))
	train_map_index = np.digitize(train_col, items, right=True)
	test_map_index = np.digitize(test_col, items, right=True)
	return newitems[train_map_index], newitems[test_map_index]

def impute(train_imp, test_imp):
	train_null_cols = pd.DataFrame()
	test_null_cols = pd.DataFrame()
	for c in train_imp.columns:
		nanrate = float(train_imp[c].isnull().sum()) / float(train_imp.shape[0])
		print(c, nanrate)
		if nanrate > 0.05:
			train_null_cols[c + '_null'] = train_imp[c].isnull()
			test_null_cols[c + '_null'] = test_imp[c].isnull()
		train_imp.ix[train_imp[c].isnull(),c] = train_imp.median()
		test_imp.ix[test_imp[c].isnull(),c] = test_imp.median()
	train_imp = pd.concat([train_imp, train_null_cols], axis=1)
	test_imp = pd.concat([test_imp, test_null_cols], axis=1)
	return train_imp, test_imp

def load(m_params):
	num_features = m_params['n_features']
	minbin = m_params['minbin']
	getcached = m_params['getcached']
	codetest = m_params['codetest']

	train = pd.read_csv('../input/train.csv')
	test  = pd.read_csv('../input/test.csv')

	if codetest:
		train = train.ix[0:30000,:]
		test = test.ix[0:30000,:]

	colsToRemove = ['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128']
	train.drop(colsToRemove, axis=1, inplace=True)
	test.drop(colsToRemove, axis=1, inplace=True)

	labels = train['target']
	test_ids = test['ID']
	train_ids = train['ID']
	train.drop(['ID','target'], axis=1, inplace=True)
	test.drop(['ID'], axis=1, inplace=True)

	print("\n\n1. Breaking dataframe into numeric, object and date parts...\n")
	train_continuous = train.select_dtypes(include=['float64'])
	test_continuous = test.select_dtypes(include=['float64'])

	print("\n\n2. Imputing ...\n")
	train_continuous, test_continuous = impute(train_continuous, test_continuous)

	train_continuous['cont_max'] = train_continuous.max(axis=1)
	train_continuous['cont_min'] = train_continuous.min(axis=1)
	train_continuous['cont_med'] = train_continuous.median(axis=1)
	train_continuous['cont_std'] = train_continuous.std(axis=1)

	test_continuous['cont_max'] = test_continuous.max(axis=1)
	test_continuous['cont_min'] = test_continuous.min(axis=1)
	test_continuous['cont_med'] = test_continuous.median(axis=1)
	test_continuous['cont_std'] = test_continuous.std(axis=1)

	train_continuous['v10-v12'] = train_continuous['v10'] - train_continuous['v12']
	test_continuous['v10-v12'] = test_continuous['v10'] - test_continuous['v12']
	train_continuous['v14-v21'] = train_continuous['v14'] - train_continuous['v21']
	test_continuous['v14-v21'] = test_continuous['v14'] - test_continuous['v21']

	test_discrete = test.select_dtypes(include=['int64'])
	train_discrete = train.select_dtypes(include=['int64'])

	train_discrete['dis_max'] = train_discrete.max(axis=1)
	train_discrete['dis_std'] = train_discrete.std(axis=1)
	train_discrete['dis_med'] = train_discrete.median(axis=1)
	
	test_discrete['dis_max'] = test_discrete.max(axis=1)
	test_discrete['dis_std'] = test_discrete.std(axis=1)
	test_discrete['dis_med'] = test_discrete.median(axis=1)

	train_categoric = train.select_dtypes(include=['object'])
	test_categoric = test.select_dtypes(include=['object'])
	
	print("\n\n2. Categorical variables...\n")
	train_categoric = train_categoric.fillna('NA')
	test_categoric = test_categoric.fillna('NA')
	
	train_categoric, test_categoric = category_transformation(train_categoric, test_categoric, labels, type = m_params['cat_trans'])

	print("train continuous shape (%s,%s)" %train_continuous.shape)
	print("train discrete shape (%s,%s)" %train_discrete.shape)
	print("train categoric shape (%s,%s)" %train_categoric.shape)

	'''
	# Deal with categorical data
	print("3. Numeric Column Smoothing... \n")

	if minbin > 1:
		for c in train_continuous.columns:
			train_continuous[c], test_continuous[c] = bin(train_continuous[c], test_continuous[c], labels, minbin)
			numeric_col_count += 1
			if not (numeric_col_count % 10):
				print('Numeric Col Count: ', numeric_col_count)

	print("\n\n3. Numeric column normalisation... \n")
	numeric_col_count = 0
	for c in train_continuous.columns:
		train_continuous[c], train_continuous[c] = np.log(train_continuous[c]+2), np.log(train_continuous[c]+2)
		numeric_col_count += 1
		if not (numeric_col_count % 10):
			print('Numeric Col Count: ', numeric_col_count)
	'''
	numeric_col_count = 0
	for c in train_discrete.columns:
		train_discrete[c], test_discrete[c] = np.log(train_discrete[c]+2), np.log(test_discrete[c]+2)
		numeric_col_count += 1
		if not (numeric_col_count % 10):
			print('Numeric Col Count: ', numeric_col_count)

	print("\n\n4. Merging arrays together...\n")
	# put seperate parts together again

	train = pd.concat([train_categoric, train_discrete, train_continuous], axis=1)
	test = pd.concat([test_categoric, test_discrete, test_continuous], axis=1)

	train = train.fillna(-2)
	test = test.fillna(-2)

	print("\n\n5. Scaling to unit variance...\n")
	scaler = StandardScaler()
	train = scaler.fit_transform(train)
	test = scaler.transform(test)

	return train, labels.values, test, test_ids.values
