import sys
import pandas as pd
import numpy as np
from time import time

from scipy import stats
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.cross_validation import StratifiedShuffleSplit, KFold
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
print('reading train data...')
client_aggs = pd.read_csv('../input/groupby_client_aggs.csv')

ids = client_aggs['ClientId']
client_aggs['TotalUnits'] = np.log1p(client_aggs['TotalUnits'])
client_aggs['CostPerUnit'] = client_aggs['TotalPesos']/client_aggs['TotalUnits']
client_aggs.drop(['TotalPesos','ClientId'], axis=1, inplace=True)
client_aggs.fillna(0,inplace=True)

scaler = StandardScaler()
client_aggs = scaler.fit_transform(client_aggs)

print("KMeans...\n")
clf1000 = KMeans(n_clusters=1000, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=1, copy_x=True, n_jobs=-1)
clf250 = KMeans(n_clusters=250, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=1, copy_x=True, n_jobs=-1)
clf50 = KMeans(n_clusters=50, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=1, copy_x=True, n_jobs=-1)

t0 = time()
c1000 = clf1000.fit_predict(client_aggs)
t1 = time()
print(t1-t0)
c250 = clf250.fit_predict(client_aggs)
t2 = time()
print(t2-t1)
c50 = clf50.fit_predict(client_aggs)
t3 = time()
print(t3-t2)

print("Saving Results.")
client_clusters = pd.DataFrame({"ClientId": ids.values, "c1000": c1000, "c250": c250, "c50": c50})
client_clusters.to_csv('../input/client_clusters.csv', index=False)
