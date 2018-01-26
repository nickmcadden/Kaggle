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
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import wordpunct_tokenize
import textmining

reload(sys)  
sys.setdefaultencoding('ISO-8859-1')
stemmer = EnglishStemmer()

# Initialize class to create term-document matrix
tdm = textmining.TermDocumentMatrix()

# Load data
print('reading product data...')
product_aggs = pd.read_csv('../input/groupby_product_aggs.csv')

'''
with open('../input/prod_tab.csv', 'r') as f1:
    lines = f1.readlines()
    print(lines[:10])
    for line in lines:
        desc = line.split(',')[6]
        # Add the documents
        tdm.add_doc(desc)

for row in tdm.rows(cutoff=30):
    print row

exit()
'''
ids = product_aggs['ProductId']
product_aggs['TotalUnits'] = np.log1p(product_aggs['TotalUnits'])
product_aggs['CostPerUnit'] = product_aggs['TotalPesos']/product_aggs['TotalUnits']
product_aggs.drop(['TotalPesos','ProductId'], axis=1, inplace=True)
product_aggs.fillna(0,inplace=True)

scaler = StandardScaler()
product_aggs = scaler.fit_transform(product_aggs)

print("KMeans...\n")
clf100 = KMeans(n_clusters=100, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=1, copy_x=True, n_jobs=-1)
clf30 = KMeans(n_clusters=30, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=1, copy_x=True, n_jobs=-1)
clf10 = KMeans(n_clusters=10, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=1, copy_x=True, n_jobs=-1)

t0 = time()
pc100 = clf100.fit_predict(product_aggs)
t1 = time()
print(t1-t0)
pc30 = clf30.fit_predict(product_aggs)
t2 = time()
print(t2-t1)
pc10 = clf10.fit_predict(product_aggs)
t3 = time()
print(t3-t2)

print("Saving Results.")
product_clusters = pd.DataFrame({"ProductId": ids.values, "pc100": pc100, "pc30": pc30, "pc10": pc10})
product_clusters.to_csv('../input/product_clusters.csv', index=False)
