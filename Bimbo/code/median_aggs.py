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

print('creating total product aggregates...')
print('cluster1000...')
prod_med = train.groupby(['c1000']).agg({'AdjDemand':np.median})
prod_med.columns = ['C1000Units']
prod_med.to_csv('../input/groupby_c1000_median_units.csv')
print('cluster250...')
prod_med = train.groupby(['c250']).agg({'AdjDemand':np.median})
prod_med.columns = ['C250Units']
prod_med.to_csv('../input/groupby_c250_median_units.csv')
print('cluster50...')
prod_med = train.groupby(['c50']).agg({'AdjDemand':np.median})
prod_med.columns = ['C50Units']
prod_med.to_csv('../input/groupby_c50_median_units.csv')
print('product...')
prod_med = train.groupby(['ProductId']).agg({'AdjDemand':np.median})
prod_med.columns = ['ProdUnits']
prod_med.to_csv('../input/groupby_product_median_units.csv')
print('cluster1000 product...')
prod_med = train.groupby(['ProductId','c1000']).agg({'AdjDemand':np.median})
prod_med.columns = ['ProdC1000Units']
prod_med.to_csv('../input/groupby_product_c1000_median_units.csv')
print('cluster250 product...')
prod_med = train.groupby(['ProductId','c250']).agg({'AdjDemand':np.median})
prod_med.columns = ['ProdC250Units']
prod_med.to_csv('../input/groupby_product_c250_median_units.csv')
print('cluster50 product...')
prod_med = train.groupby(['ProductId','c50']).agg({'AdjDemand':np.median})
prod_med.columns = ['ProdC50Units']
prod_med.to_csv('../input/groupby_product_c50_median_units.csv')
print('client product...')
prod_client_med = train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.median})
prod_client_med.columns = ['ProdClientUnits']
prod_client_med.to_csv('../input/groupby_product_client_median_units.csv')
print('client product recent week...')
prod_client_med = train[train['WeekNum']==9].groupby(['ProductId','ClientId']).agg({'AdjDemand':np.median})
prod_client_med.columns = ['ProdClientLastWeekUnits']
prod_client_med.to_csv('../input/groupby_product_client_last_week_median_units.csv')
print('route product...')
prod_route_med = train.groupby(['ProductId','RouteId']).agg({'AdjDemand':np.median})
prod_route_med.columns = ['ProdRouteUnits']
prod_route_med.to_csv('../input/groupby_product_route_median_units.csv')
print('depot product...')
prod_depot_med = train.groupby(['ProductId','DepotId']).agg({'AdjDemand':np.median})
prod_depot_med.columns = ['ProdDepotUnits']
prod_depot_med.to_csv('../input/groupby_product_depot_median_units.csv')
print('client 1...')
client_med = train.groupby(['ClientId']).agg({'AdjDemand':np.median})
client_med.columns = ['ClientUnits']
client_med.to_csv('../input/groupby_client_median_units.csv')
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
print('cluster1000...')
prod_med = train.groupby(['c1000']).agg({'AdjDemand':np.median})
prod_med.columns = ['C1000Units']
prod_med.to_csv('../input/cv/groupby_c1000_median_units.csv')
print('cluster250...')
prod_med = train.groupby(['c250']).agg({'AdjDemand':np.median})
prod_med.columns = ['C250Units']
prod_med.to_csv('../input/cv/groupby_c250_median_units.csv')
print('cluster50...')
prod_med = train.groupby(['c50']).agg({'AdjDemand':np.median})
prod_med.columns = ['C50Units']
prod_med.to_csv('../input/cv/groupby_c50_median_units.csv')
print('product...')
prod_med = train.groupby(['ProductId']).agg({'AdjDemand':np.median})
prod_med.columns = ['ProdUnits']
prod_med.to_csv('../input/cv/groupby_product_median_units.csv')
print('cluster1000 product...')
prod_med = train.groupby(['ProductId','c1000']).agg({'AdjDemand':np.median})
prod_med.columns = ['ProdC1000Units']
prod_med.to_csv('../input/cv/groupby_product_c1000_median_units.csv')
print('cluster250 product...')
prod_med = train.groupby(['ProductId','c250']).agg({'AdjDemand':np.median})
prod_med.columns = ['ProdC250Units']
prod_med.to_csv('../input/cv/groupby_product_c250_median_units.csv')
print('cluster50 product...')
prod_med = train.groupby(['ProductId','c50']).agg({'AdjDemand':np.median})
prod_med.columns = ['ProdC50Units']
prod_med.to_csv('../input/cv/groupby_product_c50_median_units.csv')
print('client product...')
prod_client_med = train.groupby(['ProductId','ClientId']).agg({'AdjDemand':np.median})
prod_client_med.columns = ['ProdClientUnits']
prod_client_med.to_csv('../input/cv/groupby_product_client_median_units.csv')
print('client product recent week...')
prod_client_med = train[train['WeekNum']==7].groupby(['ProductId','ClientId']).agg({'AdjDemand':np.median})
prod_client_med.columns = ['ProdClientLastWeekUnits']
prod_client_med.to_csv('../input/cv/groupby_product_client_last_week_median_units.csv')
print('route product...')
prod_route_med = train.groupby(['ProductId','RouteId']).agg({'AdjDemand':np.median})
prod_route_med.columns = ['ProdRouteUnits']
prod_route_med.to_csv('../input/cv/groupby_product_route_median_units.csv')
print('depot product...')
prod_depot_med = train.groupby(['ProductId','DepotId']).agg({'AdjDemand':np.median})
prod_depot_med.columns = ['ProdDepotUnits']
prod_depot_med.to_csv('../input/cv/groupby_product_depot_median_units.csv')
print('client 1...')
client_med = train.groupby(['ClientId']).agg({'AdjDemand':np.median})
client_med.columns = ['ClientUnits']
client_med.to_csv('../input/cv/groupby_client_median_units.csv')
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