import sys
import pandas as pd
import numpy as np
import data
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle

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
	'''
	print('making lag columns...')
	lag1 = traincopy[['WeekNum', 'ClientId', 'ProductId', 'AdjDemand']]
	lag1.columns = ['WeekNum', 'ClientId', 'ProductId', 'LagDemand1']
	lag1.ix[:, 'WeekNum'] += 1
	train = pd.merge(train, lag1, how='left', on=["WeekNum","ProductId","ClientId"])
	test = pd.merge(test, lag1, how='left', on=["WeekNum","ProductId","ClientId"])
	
	lag2 = traincopy[['WeekNum', 'ClientId', 'ProductId', 'AdjDemand']]
	lag2.columns = ['WeekNum', 'ClientId', 'ProductId', 'LagDemand2']
	lag2.ix[:, 'WeekNum'] += 1
	train = pd.merge(train, lag2, how='left', on=["WeekNum","ProductId","ClientId"])
	test = pd.merge(test, lag2, how='left', on=["WeekNum","ProductId","ClientId"])
	'''
	print('product client...')
	prod_client_mean =  traincopy.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean}).reset_index()
	prod_client_mean.columns = ['ProductId','ClientId','ProdClientUnits']

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

	print('merging aggregates with train...')
	train = pd.merge(train, prod_client_mean, how='left', on=["ProductId","ClientId"])
	train = pd.merge(train, client_route_mean_units, how='left', on=["ClientId","RouteId"])
	train = pd.merge(train, client_total_units, how='left', on="ClientId")
	train = pd.merge(train, client_total_pesos, how='left', on="ClientId")

	print('merging aggregates with test...')
	test = pd.merge(test, prod_client_mean, how='left', on=["ProductId","ClientId"])
	test = pd.merge(test, client_route_mean_units, how='left', on=["ClientId","RouteId"])
	test = pd.merge(test, client_total_units, how='left', on="ClientId")
	test = pd.merge(test, client_total_pesos, how='left', on="ClientId")

	global_median = np.median(train['AdjDemand'])

	trainlabels = train['AdjDemand']
	test_ids = test['id']
	train.drop(['SalesUnitsWeek', 'SalesPesosWeek','ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand','DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId'], axis=1, inplace=True)
	test.drop(['id', 'DepotId', 'ChannelId', 'RouteId', 'ClientId', 'ProductId'], axis=1, inplace=True)

	train.fillna(-1, inplace=True)
	test.fillna(-1, inplace=True)

	print(test[:100])
	return train.values, trainlabels.values, test.values, test_ids.values
