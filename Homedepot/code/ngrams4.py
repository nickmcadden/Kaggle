import xgboost as xgb
import pandas as pd
import numpy as np
import re
import time
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.util import ngrams
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

print("Reading data\n")
train = pd.read_csv('../input/train_stemmed_snowball.csv', encoding="ISO-8859-1")
test = pd.read_csv('../input/test_stemmed_snowball.csv', encoding="ISO-8859-1")
desc = pd.read_csv('../input/desc_stemmed_snowball.csv', encoding="ISO-8859-1")

train = pd.merge(train, desc, left_on = "product_uid", right_on= "product_uid", how="left", sort=False)
test = pd.merge(test, desc, left_on = "product_uid", right_on= "product_uid", how="left", sort=False)

train = shuffle(train)

t0 = time.time()
stemmer = EnglishStemmer()

def homedepot_dict_lookup(train):
	t = []
	train = pd.DataFrame(train)
	for i in train.index:
		for j in train.ix[i,3].split():
			#t.append((stemmer.stem(j.lower()), train.ix[i,4]))
			t.append((j, train.ix[i,4]))

	t = pd.DataFrame(t)
	v = pd.DataFrame(t[0].value_counts())

	t = pd.merge(t, v, left_on = 0, right_index= True)
	t = t.ix[:,1:4]
	t.columns = ['word','score','freq']

	lookup = t.groupby(by='word').mean()
	return lookup

def get_term_score_freq(words):
	words = wordpunct_tokenize(words)
	term_score = 0
	term_freq = 0
	for word in words:
		try:
			term_score += lookup.ix[word,'score'] * len(word)
			term_freq += lookup.ix[word,'freq']
		except:
			term_score += np.mean(lookup['score']) * len(word)
			term_freq += np.mean(lookup['freq'])
	term_score /= len(words)
	term_freq /= len(words)
	return term_score, term_freq

def word_match(words, title, desc):
	n1gram_title = 0
	n1gram_desc = 0
	n1keypos_match = 0
	n2gram_title = 0
	n2gram_desc = 0
	n2keypos_match = 0
	n3gram_title = 0
	n3gram_desc = 0
	n3keypos_match = 0
	brand_match = 0
	title_num_match = 0
	freqword = 0
	joining_word = 0
	total_match = 0

	search_chars = len(words)

	title = title.replace(" in . ", " inch ")
	#words = words.replace(" inch ", " in ")
	words = words.replace(" in . ", " inch ")
	words = words.replace(" ft . ", " ft ")
	#replace words seperated by backslash with appropriate spaces to allow word seperation
	words = re.sub('([a-zA-Z])/([a-zA-Z])', '\1 / \2', words)
	#split items specifying dimensions eg 4x4 in to individual components eg 4 x 4
	words = re.sub('([0-9])x([0-9])', '\1 x \2', words)

	words = wordpunct_tokenize(words)
	title = wordpunct_tokenize(title)
	desc = wordpunct_tokenize(desc)
	
	for topword in ['a','to','for','of','on','the','from']:
		for word in words:
			if word == topword:
				joining_word += 1

	search_numbers = len([int(s) for s in words if s.isdigit()])
	title_numbers = len([int(s) for s in title if s.isdigit()])
	desc_numbers = len([int(s) for s in desc if s.isdigit()])

	for n in range(1, len(words)+1):
		s_ngrams = ngrams(words, n)
		key_pos = 0
		#loop through the ngrams, key_pos will be the number of ngrams or else the one before the first that contains 'with'
		for t_ngram in ngrams(title, n):
			key_pos += 1
			if t_ngram[-1] in ['with','and','(','for','in']:
				key_pos -= 1
				break
		for s_ngram in s_ngrams:
			t_pos = 0
			for t_ngram in ngrams(title, n):
				t_pos += 1
				if s_ngram == t_ngram:
					if n == 1:
						n1gram_title += 1
						if t_pos == key_pos:
							n1keypos_match += 1
					elif n == 2:
						n2gram_title += 1
						if t_pos == key_pos:
							n2keypos_match += 1
					elif n >= 3:
						n3gram_title += 1
						if t_pos == key_pos:
							n3keypos_match += 1
					if t_pos == 1:
						brand_match = 1
					if n == len(words):
						total_match = 1
					break

			for d_ngram in ngrams(desc, n):
				if s_ngram == d_ngram:
					if n == 1:
						n1gram_desc += 1
					elif n == 2:
						n2gram_desc += 1
					elif n >= 3:
						n3gram_desc += 1
					break

	return n1gram_title, n1gram_desc, n2gram_title, n2gram_desc, n3gram_title, n3gram_desc, n1keypos_match, n2keypos_match, n3keypos_match, len(words), len(title), len(desc), search_chars, brand_match, title_numbers, desc_numbers, search_numbers, title_num_match, total_match, joining_word

print("Get number of words and word matching title in train\n")
print(train.shape)
#train_summary_stats = train.apply(lambda row: word_match(row['search_term'], row['product_title'], row['product_description']), axis=1)
#test_summary_stats = test.apply(lambda row: word_match(row['search_term'], row['product_title'], row['product_description']), axis=1)
#X = np.array(pd.DataFrame.from_records(train_summary_stats.tolist()))
#X_test = pd.DataFrame.from_records(test_summary_stats.tolist())
y = train['relevance'].values

xgb_param = {'silent' : 1, 'eta': 0.1, 'objective': 'count:poisson', 'min_child_weight': 3, 'colsample_bytree': 0.9}

# do cross validation scoring
kf = KFold(train.shape[0], n_folds=4, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])

print("Creating main features\n")
train_summary_stats = train.apply(lambda row: word_match(row['search_term'], row['product_title'], row['product_description']), axis=1)

for i, (tr_ix, val_ix) in enumerate(kf):
	# get custom lookup for this fold
	print("CV fold: %d\n" %i)
	print("Creating dictionary lookup")
	lookup = homedepot_dict_lookup(train.values[tr_ix])
	print ("Mean word score: %f\n" % np.mean(lookup['score']))
	print("Creating fold specific features\n")
	X = pd.DataFrame.from_records(train_summary_stats.tolist())

	term_score_freq = train.apply(lambda row: get_term_score_freq(row['search_term']), axis=1)
	X_append = pd.DataFrame.from_records(term_score_freq.tolist())
	X = pd.concat([X, X_append], axis=1)

 	X = np.array(X)
	dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix], missing=np.nan)
	dval = xgb.DMatrix(X[val_ix], y[val_ix], missing=np.nan)
	clf = xgb.train(xgb_param, dtrain, 200, evals=([dtrain,'train'], [dval,'val']))
	pred = clf.predict(dval)
	pred = np.clip(pred,1,3)
	scr[i] = np.sqrt(mean_squared_error(np.array(pred), y[val_ix]))
	print('CV score is: %f in %d secs' %(scr[i], time.time()-t0))
	t0 = time.time()

print('Avg score is:', np.mean(scr))
