import sys
import pandas as pd
import numpy as np
import data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle

def prob_weight_lookup(code, lookup, labels):
	try:
		scr = np.random.normal(loc=lookup.ix[code, 'tgt_rate_adj'], scale=lookup.ix[code, 'ci'])
	except:
		scr = np.mean(labels)
	return np.clip(scr,0,1)

def category_to_prob_weight(train, test, col, labels):

	traincol, testcol, labels = pd.Series(train[col]), pd.Series(test[col]), pd.Series(labels)
	kf = StratifiedKFold(labels, n_folds=5, shuffle=True, random_state=1)
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

def get_demand(key, global_median, prod_client_med, prod_route_med, prod_depot_med, prod_med, client_med):
	key = tuple(key)
	try:
		val = prod_client_med['AdjDemand'][key[:2]]
	except:
		try:
			val = prod_route_med['AdjDemand'][key[:3:2]]
		except:
			try:
				val = prod_depot_med['AdjDemand'][key[:4:3]]
			except:
				try:
					val = prod_med['AdjDemand'][key[0]]
				except:
					try:
						val = client_med['AdjDemand'][key[4]]
					except:
						val = global_median
	return val

def load(m_params):
	num_features = m_params['n_features']
	minbin = m_params['minbin']
	getcached = m_params['getcached']
	codetest = m_params['codetest']

	print('reading train data...')
	train = pd.read_hdf('../input/train.h5', 'train')
	print('reading test data...')
	test  = pd.read_hdf('../input/test.h5','test')

	if codetest:
		train = train.ix[0:29999,:]
		#test = test.ix[0:29999,:]
	
	print('shuffling train data...')
	#train = shuffle(train, random_state=1)

	train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
    	             'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
		             'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

	test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId']

	print('creating aggregates...')
	prod_med = train.groupby(['ProductId']).agg({'AdjDemand':np.median})
	print('client...')
	prod_client_med = train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.median})
	print('route...')
	prod_route_med = train.groupby(['ProductId','RouteId']).agg({'AdjDemand':np.median})
	print('depot...')
	prod_depot_med = train.groupby(['ProductId','DepotId']).agg({'AdjDemand':np.median})
	print('channel..')
	client_med = train.groupby(['ClientId']).agg({'AdjDemand':np.median})
	
	
	global_median = np.median(train['AdjDemand'])
	prod_med = prod_med.to_dict()
	prod_client_med = prod_client_med.to_dict()
	prod_route_med = prod_route_med.to_dict()
	prod_depot_med = prod_depot_med.to_dict()
	client_med = client_med.to_dict()

	labels = train['AdjDemand']
	test_ids = test['id']
	train.drop(['SalesUnitsWeek', 'SalesPesosWeek','ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand'], axis=1, inplace=True)
	test.drop(['id'], axis=1, inplace=True)

	train = train.fillna(0)
	test = test.fillna(0)

	#Generating the output
	print('building results...')
	pred = test[['ProductId', 'ClientId', 'RouteId','DepotId','ChannelId']].apply(lambda x:get_demand(x, global_median, prod_client_med, prod_route_med, prod_depot_med, prod_med, client_med), axis=1)
	print("Saving Results.")
	preds = pd.DataFrame({"id": test_ids, "Demanda_uni_equil": pred})
	print(preds[:100])
	preds.to_csv('../output/product_client_median.csv', index=False)

	exit()
	return train, labels.values, test, test_ids.values
