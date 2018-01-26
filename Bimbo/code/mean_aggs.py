#!/usr/bin/env python
# encoding: utf-8
"""
data2.py

Created by Nick McAdden on 2016-06-18.
Copyright (c) 2016 __MyCompanyName__. All rights reserved.
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer as DV
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle

print('reading train data...')
train = pd.read_hdf('../input/train.h5', 'train')
print('reading test data...')
test  = pd.read_hdf('../input/test.h5','test')

print('shuffling train data...')
#train = shuffle(train, random_state=1)

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

print('creating total product aggregates...')
print('product cluster100...')
prod_mean = train.groupby(['pc100']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['PC100Units']
prod_mean.to_csv('../input/groupby_pc100_mean_units.csv')
print('product cluster30...')
prod_mean = train.groupby(['pc30']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['PC30Units']
prod_mean.to_csv('../input/groupby_pc30_mean_units.csv')
print('product cluster10...')
prod_mean = train.groupby(['pc10']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['PC10Units']
prod_mean.to_csv('../input/groupby_pc10_mean_units.csv')
print('product...')
prod_mean = train.groupby(['ProductId']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['ProdUnits']
prod_mean.to_csv('../input/groupby_product_mean_units.csv')
print('cluster1000 product...')
prod_mean = train.groupby(['ProductId','c1000']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['ProdC1000Units']
prod_mean.to_csv('../input/groupby_product_c1000_mean_units.csv')
print('cluster250 product...')
prod_mean = train.groupby(['ProductId','c250']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['ProdC250Units']
prod_mean.to_csv('../input/groupby_product_c250_mean_units.csv')
print('cluster50 product...')
prod_mean = train.groupby(['ProductId','c50']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['ProdC50Units']
prod_mean.to_csv('../input/groupby_product_c50_mean_units.csv')
print('client product agent route...')
prod_client_mean = train.groupby(['ProductId','ClientId','DepotId','RouteId']).agg({'AdjDemand':np.mean})
prod_client_mean.columns = ['ProdClientAgentRouteUnits']
prod_client_mean.to_csv('../input/groupby_product_client_agent_route_mean_units.csv')
print('client product agent...')
prod_client_mean = train.groupby(['ProductId','ClientId','DepotId']).agg({'AdjDemand':np.mean})
prod_client_mean.columns = ['ProdClientAgentUnits']
prod_client_mean.to_csv('../input/groupby_product_client_agent_mean_units.csv')
print('client product...')
prod_client_mean = train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean})
prod_client_mean.columns = ['ProdClientUnits']
prod_client_mean.to_csv('../input/groupby_product_client_mean_units.csv')
print('client product recent week...')
prod_client_mean = train[train['WeekNum']==9].groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean})
prod_client_mean.columns = ['ProdClientLastWeekUnits']
prod_client_mean.to_csv('../input/groupby_product_client_last_week_mean_units.csv')
print('route product...')
prod_route_mean = train.groupby(['ProductId','RouteId']).agg({'AdjDemand':np.mean})
prod_route_mean.columns = ['ProdRouteUnits']
prod_route_mean.to_csv('../input/groupby_product_route_mean_units.csv')
print('depot product...')
prod_depot_mean = train.groupby(['ProductId','DepotId']).agg({'AdjDemand':np.mean})
prod_depot_mean.columns = ['ProdDepotUnits']
prod_depot_mean.to_csv('../input/groupby_product_depot_mean_units.csv')
print('client 1...')
client_mean = train.groupby(['ClientId']).agg({'AdjDemand':np.mean})
client_mean.columns = ['ClientUnits']
client_mean.to_csv('../input/groupby_client_mean_units.csv')
print('client 2...')
client_total_units = train.groupby(['ClientId']).agg({'AdjDemand':np.sum})
client_total_units.columns = ['ClientTotalUnits']
client_total_units.to_csv('../input/groupby_client_total_units.csv')
print('client 3...')
client_total_pesos = train.groupby(['ClientId']).agg({'SalesPesosWeek':np.sum})
client_total_pesos.columns = ['ClientTotalPesos']
client_total_pesos.to_csv('../input/groupby_client_total_pesos.csv')

train = train[train['WeekNum']<=7]

print('creating cv aggregates...')
print('product cluster100...')
prod_mean = train.groupby(['pc100']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['PC100Units']
prod_mean.to_csv('../input/cv/groupby_pc100_mean_units.csv')
print('product cluster30...')
prod_mean = train.groupby(['pc30']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['PC30Units']
prod_mean.to_csv('../input/cv/groupby_pc30_mean_units.csv')
print('product cluster10...')
prod_mean = train.groupby(['pc10']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['PC10Units']
prod_mean.to_csv('../input/cv/groupby_pc10_mean_units.csv')
print('product...')
prod_mean = train.groupby(['ProductId']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['ProdUnits']
prod_mean.to_csv('../input/cv/groupby_product_mean_units.csv')
print('cluster1000 product...')
prod_mean = train.groupby(['ProductId','c1000']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['ProdC1000Units']
prod_mean.to_csv('../input/cv/groupby_product_c1000_mean_units.csv')
print('cluster250 product...')
prod_mean = train.groupby(['ProductId','c250']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['ProdC250Units']
prod_mean.to_csv('../input/cv/groupby_product_c250_mean_units.csv')
print('cluster50 product...')
prod_mean = train.groupby(['ProductId','c50']).agg({'AdjDemand':np.mean})
prod_mean.columns = ['ProdC50Units']
prod_mean.to_csv('../input/cv/groupby_product_c50_mean_units.csv')
print('client product agent route...')
prod_client_mean = train.groupby(['ProductId','ClientId','DepotId','RouteId']).agg({'AdjDemand':np.mean})
prod_client_mean.columns = ['ProdClientAgentRouteUnits']
prod_client_mean.to_csv('../input/cv/groupby_product_client_agent_route_mean_units.csv')
print('client product agent...')
prod_client_mean = train.groupby(['ProductId','ClientId','DepotId']).agg({'AdjDemand':np.mean})
prod_client_mean.columns = ['ProdClientAgentUnits']
prod_client_mean.to_csv('../input/cv/groupby_product_client_agent_mean_units.csv')
print('client product...')
prod_client_mean = train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean})
prod_client_mean.columns = ['ProdClientUnits']
prod_client_mean.to_csv('../input/cv/groupby_product_client_mean_units.csv')
print('client product recent week...')
prod_client_mean = train[train['WeekNum']==7].groupby(['ProductId','ClientId']).agg({'AdjDemand':np.mean})
prod_client_mean.columns = ['ProdClientLastWeekUnits']
prod_client_mean.to_csv('../input/cv/groupby_product_client_last_week_mean_units.csv')
print('route product...')
prod_route_mean = train.groupby(['ProductId','RouteId']).agg({'AdjDemand':np.mean})
prod_route_mean.columns = ['ProdRouteUnits']
prod_route_mean.to_csv('../input/cv/groupby_product_route_mean_units.csv')
print('depot product...')
prod_depot_mean = train.groupby(['ProductId','DepotId']).agg({'AdjDemand':np.mean})
prod_depot_mean.columns = ['ProdDepotUnits']
prod_depot_mean.to_csv('../input/cv/groupby_product_depot_mean_units.csv')
print('client 1...')
client_mean = train.groupby(['ClientId']).agg({'AdjDemand':np.mean})
client_mean.columns = ['ClientUnits']
client_mean.to_csv('../input/cv/groupby_client_mean_units.csv')
print('client 2...')
client_total_units = train.groupby(['ClientId']).agg({'AdjDemand':np.sum})
client_total_units.columns = ['ClientTotalUnits']
client_total_units.to_csv('../input/cv/groupby_client_total_units.csv')
print('client 3...')
client_total_pesos = train.groupby(['ClientId']).agg({'SalesPesosWeek':np.sum})
client_total_pesos.columns = ['ClientTotalPesos']
client_total_pesos.to_csv('../input/cv/groupby_client_total_pesos.csv')

'''
print('creating client aggegates...')
client_aggs = train.groupby(['ClientId']).agg({'AdjDemand':np.sum, 'SalesPesosWeek':np.sum, 'ProductId':pd.Series.nunique})
client_aggs.columns = ['TotalUnits','TotalPesos','DistinctProducts']
client_aggs.to_csv('../input/groupby_client_aggs.csv')
'''