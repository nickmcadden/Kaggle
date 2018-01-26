import pandas as pd
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import wordpunct_tokenize
import sys
import csv

reload(sys)  
sys.setdefaultencoding('ISO-8859-1')
stemmer = EnglishStemmer()

print("Reading data\n")
train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")
desc = pd.read_csv('./input/product_descriptions.csv', encoding="ISO-8859-1")

print("Stemming train file\n")
for index, row in train.iterrows():
	train.ix[index,'product_title'] = " ".join([stemmer.stem(word.lower()) for word in wordpunct_tokenize(row['product_title'])])
	train.ix[index,'search_term'] = " ".join([stemmer.stem(word.lower()) for word in wordpunct_tokenize(row['search_term'])])
	if index % 1000 == 0:
		print(index)

train.to_csv('./input/train_stemmed_snowball.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

print("\nStemming test file\n")
for index, row in test.iterrows():
	test.ix[index,'product_title'] = " ".join([stemmer.stem(word.lower()) for word in wordpunct_tokenize(row['product_title'])])
	test.ix[index,'search_term'] = " ".join([stemmer.stem(word.lower()) for word in wordpunct_tokenize(row['search_term'])])
	if index % 1000 == 0:
		print(index)

test.to_csv('./input/test_stemmed_snowball.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
'''
print("\nStemming description file\n")
for index, row in desc.iterrows():
	desc.ix[index,'product_description'] = " ".join([stemmer.stem(word.lower()) for word in wordpunct_tokenize(row['product_description'])])
	if index % 1000 == 0:
		print(index)

desc.to_csv('./input/desc_stemmed_snowball.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
'''

