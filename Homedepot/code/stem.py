import pandas as pd
import numpy as np
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import wordpunct_tokenize
import sys

reload(sys)  
sys.setdefaultencoding('ISO-8859-1')
stemmer = EnglishStemmer()

with open('./input/train.csv', mode='r') as f1:
   with open('./input/train_stemmed.csv', 'w') as f2:
       lines = f1.readlines()
       for i, line in enumerate(lines):
          f2.write(" ".join([stemmer.stem(word.lower()) for word in wordpunct_tokenize(line)]) + "\n")


with open('./input/test.csv', mode='r') as f1:
   with open('./input/test_stemmed.csv', 'w') as f2:
       lines = f1.readlines()
       for i, line in enumerate(lines):
          f2.write(" ".join([stemmer.stem(word.lower()) for word in wordpunct_tokenize(line)]) + "\n")


with open('./input/product_descriptions.csv', mode='r') as f1:
   with open('./input/product_descriptions_stemmed.csv', 'w') as f2:
       lines = f1.readlines()
       for i, line in enumerate(lines):
          f2.write(" ".join([stemmer.stem(word.lower()) for word in wordpunct_tokenize(line)]) + "\n")
