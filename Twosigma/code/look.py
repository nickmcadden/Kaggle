import pandas as pd
import numpy as np
from collections import Counter
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import wordpunct_tokenize
import sys

print("Reading data\n")
train = pd.read_json('../input/train.json', encoding="ISO-8859-1").reindex()

train["latlon"] = np.abs(train["latitude"] - train["longitude"])

for i in train.columns:
	print(i, train[i].dtype)
	print('unique', len(np.unique(train[i])))
	if len(np.unique(train[i]))<30:
		print(train[i].value_counts())
	print('null', np.sum(pd.isnull(train[i])))
	print(train[i][:10])

exit()

print(train.head)
print(train.shape)

stemmer = EnglishStemmer()

for i in range(train.shape[0]-1):
	if train.ix[i,'relevance'] == train.ix[i+1,'relevance'] and train.ix[i,'search_term'] == train.ix[i+1,'search_term']:
		print(i)

t = []
for i in train.index:
	for j in wordpunct_tokenize(train.ix[i,'search_term']):
		t.append((stemmer.stem(j.lower()), train.ix[i,'relevance']))
		#t.append((j.lower(), train.ix[i,'relevance']))

t = pd.DataFrame(t)
v = pd.DataFrame(t[0].value_counts())

t = pd.merge(t, pd.DataFrame(v), left_on = 0, right_index= True)
t = t.ix[:,1:4]
t.columns = ['a','b','c']

t = t.groupby(by='a').mean().sort('c', ascending=False)

with pd.option_context('display.max_rows', 2000):
	print(t)
	

