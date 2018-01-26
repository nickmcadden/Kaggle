import sys
import pandas as pd
import numpy as np
import data
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

def load(m_params):
	num_features = m_params['n_features']
	minbin = m_params['minbin']
	getcached = m_params['getcached']
	codetest = m_params['codetest']
	cv = m_params['cv']

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
	
	product_clusters  = pd.read_csv('../input/product_clusters.csv')
	train = pd.merge(train, product_clusters, how='left', on="ProductId")
	test = pd.merge(test, product_clusters, how='left', on="ProductId")

	traincopy = train.copy()
	
	if codetest:
		train = train.iloc[0:1000000,:]
		#test = test.ix[0:29999,:]

	if cv:
		inputdir = '../input/cv/'
		traincopy = traincopy[traincopy['WeekNum']<=7]
	else:
		inputdir = '../input/'

	traincopy['AdjDemand'] = traincopy['SalesUnitsWeek']

	print('reading aggregates...')
	prod_mean =  traincopy.groupby(['ProductId']).agg({'AdjDemand':np.mean}).reset_index()
	prod_mean.columns = ['ProductId','ProdUnits']
	print('product client route...')
	prod_client_route_mean =  traincopy.groupby(['ProductId','ClientId','RouteId']).agg({'AdjDemand':np.mean}).reset_index()
	prod_client_route_mean.columns = ['ProductId','ClientId','RouteId','ProdClientRouteUnits']
	print('product client...')
	prod_client_mean =  traincopy.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean}).reset_index()
	prod_client_mean.columns = ['ProductId','ClientId','ProdClientUnits']
	print('product route...')
	prod_route_mean = traincopy.groupby(['ProductId','RouteId']).agg({'AdjDemand':np.mean}).reset_index()
	prod_route_mean.columns = ['ProductId','RouteId','ProdRouteUnits']
	print('product depot...')
	prod_depot_mean = traincopy.groupby(['ProductId','DepotId']).agg({'AdjDemand':np.mean}).reset_index()
	prod_depot_mean.columns = ['ProductId','DepotId','ProdDepotUnits']
	print('client mean units..')
	client_mean_units = traincopy.groupby(['ClientId']).agg({'AdjDemand':np.mean}).reset_index()
	client_mean_units.columns = ['ClientId','ClientUnits']
	print('client route units..')
	client_route_mean_units = traincopy.groupby(['ClientId','RouteId']).agg({'AdjDemand':np.mean}).reset_index()
	client_route_mean_units.columns = ['ClientId','RouteId','ClientRouteUnits']
	print('client total units..')
	client_total_units =  traincopy.groupby(['ClientId']).agg({'AdjDemand':np.sum}).reset_index()
	client_total_units.columns = ['ClientId','ClientTotalUnits']
	print('client total pesos..')
	client_total_pesos =  traincopy.groupby(['ClientId']).agg({'SalesPesosWeek':np.sum}).reset_index()
	client_total_pesos.columns = ['ClientId','ClientTotalPesos']

	print('creating product aggegates...')
	product_aggs = train.groupby(['ProductId']).agg({'AdjDemand':np.sum, 'SalesPesosWeek':np.sum, 'ClientId':pd.Series.nunique}).reset_index()
	product_aggs.columns = ['ProductId','TotalUnits','TotalPesos','DistinctClients']
	product_aggs.to_csv('../input/groupby_product_aggs.csv')

	print('prod 100 client 1000 cluster..')
	pc100_c1000_mean =  traincopy.groupby(['pc100', 'c1000']).agg({'AdjDemand':np.mean}).reset_index()
	pc100_c1000_mean.columns = ['pc100', 'c1000', 'C1000Units']
	print('prod 20 client 250 cluster..')
	pc30_c250_mean =  traincopy.groupby(['pc30', 'c250']).agg({'AdjDemand':np.mean}).reset_index()
	pc30_c250_mean.columns = ['pc30', 'c250', 'C250Units']
	print('prod 10 client 50 cluster..')
	pc10_c50_mean =  traincopy.groupby(['pc10', 'c50']).agg({'AdjDemand':np.mean}).reset_index()
	pc10_c50_mean.columns = ['pc10', 'c50', 'C50Units']

	print('product c1000 cluster..')
	prod_c1000_mean =  traincopy.groupby(['ProductId','c1000']).agg({'AdjDemand':np.mean}).reset_index()
	prod_c1000_mean.columns = ['ProductId','c1000','ProdC1000Units']
	print('product c250 cluster..')
	prod_c250_mean =  traincopy.groupby(['ProductId','c250']).agg({'AdjDemand':np.mean}).reset_index()
	prod_c250_mean.columns = ['ProductId','c250','ProdC250Units']
	print('product c50 cluster..')
	prod_c50_mean = traincopy.groupby(['ProductId','c50']).agg({'AdjDemand':np.mean}).reset_index()
	prod_c50_mean.columns = ['ProductId','c50','ProdC50Units']

	print('merging aggregates with train...')
	#train = pd.merge(train, prod_detail, how='left', on="ProductId")
	#train = pd.merge(train, prod, how='left', on="ProductId")
	train = pd.merge(train, prod_mean, how='left', on="ProductId")
	train = pd.merge(train, prod_client_route_mean, how='left', on=["ProductId","ClientId","RouteId"])
	train = pd.merge(train, prod_client_mean, how='left', on=["ProductId","ClientId"])
	train = pd.merge(train, prod_c1000_mean, how='left', on=["ProductId","c1000"])
	train = pd.merge(train, prod_c250_mean, how='left', on=["ProductId","c250"])
	train = pd.merge(train, prod_c50_mean, how='left', on=["ProductId","c50"])
	train = pd.merge(train, pc100_c1000_mean, how='left', on=["pc100", "c1000"])
	train = pd.merge(train, pc30_c250_mean, how='left', on=["pc30", "c250"])
	train = pd.merge(train, pc10_c50_mean, how='left', on=["pc10", "c50"])
	train = pd.merge(train, prod_route_mean, how='left', on=["ProductId","RouteId"])
	train = pd.merge(train, prod_depot_mean, how='left', on=["ProductId","DepotId"])
	train = pd.merge(train, client_mean_units, how='left', on="ClientId")
	train = pd.merge(train, client_route_mean_units, how='left', on=["ClientId","RouteId"])
	train = pd.merge(train, client_total_units, how='left', on="ClientId")
	train = pd.merge(train, client_total_pesos, how='left', on="ClientId")
	
	print('merging aggregates with test...')
	#train = pd.merge(train, prod_detail, how='left', on="ProductId")
	#train = pd.merge(train, prod, how='left', on="ProductId")
	test = pd.merge(test, prod_mean, how='left', on="ProductId")
	test = pd.merge(test, prod_client_route_mean, how='left', on=["ProductId","ClientId","RouteId"])
	test = pd.merge(test, prod_client_mean, how='left', on=["ProductId","ClientId"])
	test = pd.merge(test, prod_c1000_mean, how='left', on=["ProductId","c1000"])
	test = pd.merge(test, prod_c250_mean, how='left', on=["ProductId","c250"])
	test = pd.merge(test, prod_c50_mean, how='left', on=["ProductId","c50"])
	test = pd.merge(test, pc100_c1000_mean, how='left', on=["pc100", "c1000"])
	test = pd.merge(test, pc30_c250_mean, how='left', on=["pc30", "c250"])
	test = pd.merge(test, pc10_c50_mean, how='left', on=["pc10", "c50"])
	test = pd.merge(test, prod_route_mean, how='left', on=["ProductId","RouteId"])
	test = pd.merge(test, prod_depot_mean, how='left', on=["ProductId","DepotId"])
	test = pd.merge(test, client_mean_units, how='left', on="ClientId")
	test = pd.merge(test, client_route_mean_units, how='left', on=["ClientId","RouteId"])
	test = pd.merge(test, client_total_units, how='left', on="ClientId")
	test = pd.merge(test, client_total_pesos, how='left', on="ClientId")

	train['hasProdClient'] = np.isfinite(train['ProdClientUnits'])
	test['hasProdClient'] = np.isfinite(test['ProdClientUnits'])
	train['hasProdClientRoute'] = np.isfinite(train['ProdClientRouteUnits'])
	test['hasProdClientRoute'] = np.isfinite(test['ProdClientRouteUnits'])

	global_median = np.median(train['AdjDemand'])

	trainlabels = train['AdjDemand']
	test_ids = test['id']
	train.drop(['SalesUnitsWeek','ProdClientRouteUnits', 'SalesPesosWeek','ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand','DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId'], axis=1, inplace=True)
	test.drop(['id','ProdClientRouteUnits', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId'], axis=1, inplace=True)

	train['ClientUnits'] = train['ClientUnits'].fillna(global_median)
	test['ClientUnits'] = test['ClientUnits'].fillna(global_median)
	train['ProdUnits'] = train['ProdUnits'].fillna(train['ClientUnits'])
	test['ProdUnits'] = test['ProdUnits'].fillna(test['ClientUnits'])
	train['ProdC50Units'] = train['ProdC50Units'].fillna(train['ProdUnits'])
	test['ProdC50Units'] = test['ProdC50Units'].fillna(train['ProdUnits'])
	train['ProdC250Units'] = train['ProdC250Units'].fillna(train['ProdC50Units'])
	test['ProdC250Units'] = test['ProdC250Units'].fillna(test['ProdC50Units'])
	train['ProdC1000Units'] = train['ProdC1000Units'].fillna(train['ProdC250Units'])
	test['ProdC1000Units'] = test['ProdC1000Units'].fillna(test['ProdC250Units'])
	train['ProdClientUnits'] = train['ProdClientUnits'].fillna(train['ProdC1000Units'])
	test['ProdClientUnits'] = test['ProdClientUnits'].fillna(test['ProdC1000Units'])
	train = train.fillna(global_median)
	test = test.fillna(global_median)

	print(test[:100])
	return train.values, trainlabels.values, test.values, test_ids.values
