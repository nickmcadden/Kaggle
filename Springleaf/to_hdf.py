import pandas as pd
import numpy as np
import gc
import tables
import time
import datetime as dt
import pickle

def dateparse(x):
	try:
		return pd.datetime.strptime(x, "%d%b%y:%H:%M:%S")
	except:
		return pd.NaT

datecols = pd.DataFrame({'col':['VAR_0156','VAR_0157','VAR_0158','VAR_0159','VAR_0166','VAR_0167','VAR_0168','VAR_0169','VAR_0176','VAR_0177','VAR_0178','VAR_0179','VAR_0073','VAR_0075','VAR_0204','VAR_0217']})
pd.to_pickle(datecols, 'datecols.pkl')
datecols = datecols['col'].values.tolist()

t0 = time.time()
print("reading the train and test data\n")
train = pd.read_csv('train.csv', parse_dates=datecols, date_parser=dateparse)
test  = pd.read_csv('test.csv', parse_dates=datecols, date_parser=dateparse)
t1 = time.time()
print(t1-t0)

print("-1 cleaning..\n")
for c in train.columns[1:-1]:
	if train[c].dtype.name == "float64":
		coltemp = train[c].values
		coltemp[coltemp == -1] = 0
		train[c] = coltemp
		coltemp = test[c].values
		coltemp[coltemp == -1] = 0
		test[c] = coltemp

'''
for c in datecols:
	train[c+'_ym']=train[c].dt.year*100+train[c].dt.month
	train[c+'_wd']=train[c].dt.weekday
	train[c+'_hr']=train[c].dt.hour
	test[c+'_ym']=test[c].dt.year*100+test[c].dt.month
	test[c+'_wd']=test[c].dt.weekday
	test[c+'_hr']=test[c].dt.hour
'''
gc.collect()

print("drop static crap columns with 1 unique value\n")	
for c in train.columns:
	if len(np.unique(train[c])) == 1:
		print(c, train[c].dtype.name, np.max(train[c]), np.min(train[c]), len(np.unique(train[c])))
		train.drop([c],axis=1,inplace=True)
		test.drop([c],axis=1,inplace=True)
		
print("writing to hdf format\n")
train.to_hdf('train.h5',key='train',format='fixed',mode='w')
test.to_hdf('test.h5',key='test',format='fixed',mode='w')
gc.collect()
