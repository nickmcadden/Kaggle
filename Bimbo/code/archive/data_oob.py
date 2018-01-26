import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle

# client
# >> similar client
# route
# depot
# >> state
# overall product
# >> product_type
# global_median

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

def load(m_params):
	num_features = m_params['n_features']
	minbin = m_params['minbin']
	getcached = m_params['getcached']
	codetest = m_params['codetest']
	cv = m_params['cv']
	cvweek = m_params['cvweek']

	print('reading train data...')
	train = pd.read_hdf('../input/train.h5', 'train')
	print('reading test data...')
	test  = pd.read_hdf('../input/test.h5','test')

	print('shuffling train data...')
	train = shuffle(train, random_state=1)

	train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
    	             'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
		             'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

	test.columns = ['id', 'WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId']
	
	print('merging clusters with train and test data')
	client_clusters  = pd.read_csv('../input/client_clusters.csv')
	train = pd.merge(train, client_clusters, how='left', on="ClientId")
	test = pd.merge(test, client_clusters, how='left', on="ClientId")

	traincopy = train.copy()
	
	if codetest:
		train = train.iloc[0:1000000,:]
		#test = test.ix[0:29999,:]

	if cv:
		print('week', cvweek)
		traincopy = traincopy[traincopy['WeekNum']!=cvweek]
		print('shape', traincopy.shape)

	#print('reading product data...')
	#prod_detail = pd.read_csv('../input/preprocessed_products.csv')
	print('reading aggregates...')
	prod_med = traincopy.groupby(['ProductId']).agg({'AdjDemand':np.median}).reset_index()
	prod_med.columns = ['ProductId','ProdUnits']
	print('product client...')
	prod_client_med = traincopy.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.median}).reset_index()
	prod_client_med.columns = ['ProductId','ClientId','ProdClientUnits']
	print('product route...')
	prod_route_med = traincopy.groupby(['ProductId','RouteId']).agg({'AdjDemand':np.median}).reset_index()
	prod_route_med.columns = ['ProductId','RouteId','ProdRouteUnits']	
	print('product depot...')
	prod_depot_med = traincopy.groupby(['ProductId','DepotId']).agg({'AdjDemand':np.median}).reset_index()
	prod_depot_med.columns = ['ProductId','DepotId','ProdDepotUnits']	
	print('client median units..')
	client_med_units = traincopy.groupby(['ClientId']).agg({'AdjDemand':np.median}).reset_index()
	client_med_units.columns = ['ClientId','ClientUnits']
	print('client total units..')
	client_total_units = traincopy.groupby(['ClientId']).agg({'AdjDemand':np.sum}).reset_index()
	client_total_units.columns = ['ClientId','ClientTotalUnits']
	print('client total pesos..')
	client_total_pesos = traincopy.groupby(['ClientId']).agg({'SalesPesosWeek':np.sum}).reset_index()
	client_total_pesos.columns = ['ClientId','ClientTotalPesos']
	print('c1000 cluster..')
	c1000_med  = traincopy.groupby(['c1000']).agg({'AdjDemand':np.median}).reset_index()
	c1000_med .columns = ['c1000','C1000Units']
	print('c250 cluster..')
	c250_med = traincopy.groupby(['c250']).agg({'AdjDemand':np.median}).reset_index()
	c250_med.columns = ['c250','C250Units']
	print('c50 cluster..')
	c50_med = traincopy.groupby(['c50']).agg({'AdjDemand':np.median}).reset_index()
	c50_med.columns = ['c50','C50Units']
	print('product c1000 cluster..')
	prod_c1000_med = traincopy.groupby(['ProductId','c1000']).agg({'AdjDemand':np.median}).reset_index()
	prod_c1000_med.columns = ['ProductId','c1000','ProdC1000Units']
	print('product c250 cluster..')
	prod_c250_med = traincopy.groupby(['ProductId','c250']).agg({'AdjDemand':np.median}).reset_index()
	prod_c250_med.columns = ['ProductId','c250','ProdC250Units']
	print('product c50 cluster..')
	prod_c50_med = traincopy.groupby(['ProductId','c50']).agg({'AdjDemand':np.median}).reset_index()
	prod_c50_med.columns = ['ProductId','c50','ProdC50Units']

	print('merging aggregates with train...')
	#train = pd.merge(train, prod_detail, how='left', on="ProductId")
	train = pd.merge(train, prod_med, how='left', on="ProductId")
	train = pd.merge(train, prod_client_med, how='left', on=["ProductId","ClientId"])
	train = pd.merge(train, prod_c1000_med, how='left', on=["ProductId","c1000"])
	train = pd.merge(train, prod_c250_med, how='left', on=["ProductId","c250"])
	train = pd.merge(train, prod_c50_med, how='left', on=["ProductId","c50"])
	#train = pd.merge(train, c1000_med, how='left', on=["c1000"])
	#train = pd.merge(train, c250_med, how='left', on=["c250"])
	#train = pd.merge(train, c50_med, how='left', on=["c50"])
	train = pd.merge(train, prod_route_med, how='left', on=["ProductId","RouteId"])
	train = pd.merge(train, prod_depot_med, how='left', on=["ProductId","DepotId"])
	train = pd.merge(train, client_med_units, how='left', on="ClientId")
	train = pd.merge(train, client_total_units, how='left', on="ClientId")
	train = pd.merge(train, client_total_pesos, how='left', on="ClientId")

	print('merging aggregates with test...')
	#test = pd.merge(test, prod_detail, how='left', on="ProductId")
	test = pd.merge(test, prod_med, how='left', on="ProductId")
	test = pd.merge(test, prod_client_med, how='left', on=["ProductId","ClientId"])
	test = pd.merge(test, prod_c1000_med, how='left', on=["ProductId","c1000"])
	test = pd.merge(test, prod_c250_med, how='left', on=["ProductId","c250"])
	test = pd.merge(test, prod_c50_med, how='left', on=["ProductId","c50"])
	test = pd.merge(test, prod_route_med, how='left', on=["ProductId","RouteId"])
	test = pd.merge(test, prod_depot_med, how='left', on=["ProductId","DepotId"])
	#test = pd.merge(test, c1000_med, how='left', on=["c1000"])
	#test = pd.merge(test, c250_med, how='left', on=["c250"])
	#test = pd.merge(test, c50_med, how='left', on=["c50"])
	test = pd.merge(test, client_med_units, how='left', on="ClientId")
	test = pd.merge(test, client_total_units, how='left', on="ClientId")
	test = pd.merge(test, client_total_pesos, how='left', on="ClientId")

	#train['ClientCostPerUnit'] = train['ClientTotalPesos']/train['ClientTotalUnits']
	#test['ClientCostPerUnit'] = test['ClientTotalPesos']/test['ClientTotalUnits']
	train['hasProdClient'] = np.isfinite(train['ProdClientUnits'])
	test['hasProdClient'] = np.isfinite(test['ProdClientUnits'])
	#train['hasProdClientLastWeek'] = np.isfinite(train['ProdClientLastWeekUnits'])
	#test['hasProdClientLastWeek'] = np.isfinite(test['ProdClientLastWeekUnits'])

	global_median = np.median(train['AdjDemand'])

	trainlabels = train['AdjDemand']
	test_ids = test['id']
	train.drop(['c1000','c250','c50','SalesUnitsWeek', 'SalesPesosWeek','ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand','DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId'], axis=1, inplace=True)
	test.drop(['id','c1000','c250','c50','DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId'], axis=1, inplace=True)

	#train['ProdClientLastWeekUnits'] = train['ProdClientLastWeekUnits'].fillna(train['ProdClientUnits'])
	#test['ProdClientLastWeekUnits'] = test['ProdClientLastWeekUnits'].fillna(test['ProdClientUnits'])
	#train['C50Units'] = train['C50Units'].fillna(global_median)
	#test['C50Units'] = test['C50Units'].fillna(global_median)
	#train['C250Units'] = train['C250Units'].fillna(train['C50Units'])
	#test['C250Units'] = test['C250Units'].fillna(test['C50Units'])
	#train['C1000Units'] = train['C1000Units'].fillna(train['C50Units'])
	#test['C1000Units'] = test['C1000Units'].fillna(test['C250Units'])
	train['ClientUnits'] = train['ClientUnits'].fillna(global_median)
	test['ClientUnits'] = test['ClientUnits'].fillna(global_median)
	train['ProdUnits'] = train['ProdUnits'].fillna(train['ClientUnits'])
	test['ProdUnits'] = test['ProdUnits'].fillna(test['ClientUnits'])
	train['ProdDepotUnits'] = train['ProdDepotUnits'].fillna(train['ProdUnits'])
	test['ProdDepotUnits'] = test['ProdDepotUnits'].fillna(test['ProdUnits'])
	train['ProdRouteUnits'] = train['ProdRouteUnits'].fillna(train['ProdDepotUnits'])
	test['ProdRouteUnits'] = test['ProdRouteUnits'].fillna(test['ProdDepotUnits'])
	train['ProdClientUnits'] = train['ProdClientUnits'].fillna(train['ProdRouteUnits'])
	test['ProdClientUnits'] = test['ProdClientUnits'].fillna(test['ProdRouteUnits'])
	train = train.fillna(global_median)
	test = test.fillna(global_median)

	print(train[:100])

	return train.values, trainlabels.values, test.values, test_ids.values