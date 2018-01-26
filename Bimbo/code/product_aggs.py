#!/usr/bin/env python
# encoding: utf-8
"""
Created by Nick McAdden on 2016-06-18.
Copyright (c) 2016 __MyCompanyName__. All rights reserved.
"""
import sys
import pandas as pd
import numpy as np

print('reading train data...')
train = pd.read_hdf('../input/train.h5', 'train')

train.columns = ['WeekNum', 'DepotId', 'ChannelId', 'RouteId', 'ClientId',\
	             'ProductId', 'SalesUnitsWeek', 'SalesPesosWeek',\
	             'ReturnsUnitsWeek', 'ReturnsPesosWeek', 'AdjDemand']

print('creating product aggegates...')
product_aggs = train.groupby(['ProductId']).agg({'AdjDemand':np.sum, 'SalesPesosWeek':np.sum, 'ClientId':pd.Series.nunique})
product_aggs.columns = ['TotalUnits','TotalPesos','DistinctClients']
product_aggs.to_csv('../input/groupby_product_aggs.csv')


