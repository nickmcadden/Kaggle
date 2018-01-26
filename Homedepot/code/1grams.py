import xgboost as xgb
import pandas as pd
import numpy as np
from nltk import stem
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.util import ngrams
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

print("Reading data\n")
train = pd.read_csv('./input/train.csv', encoding="ISO-8859-1")
test = pd.read_csv('./input/test.csv', encoding="ISO-8859-1")
desc = pd.read_csv('./input/product_descriptions.csv', encoding="ISO-8859-1")

train = pd.merge(train, desc, left_on = "product_uid", right_on= "product_uid", how="left", sort=False)
test = pd.merge(test, desc, left_on = "product_uid", right_on= "product_uid", how="left", sort=False)

stemmer = PorterStemmer()

def word_match(words, title, desc):
	n_title = 0
	n_desc = 0
	brand_match = 0
	last_match = 0
	title_num_match = 0
	freqword = 0
	search_chars = len(words)

	words = words.replace(" inch ", " in ")
	words = words.replace(" in. ", " in ")
	words = words.replace(" ft. ", " ft ")
	words = words.replace("' ", " ft ")
	
	words = wordpunct_tokenize(words)
	title = wordpunct_tokenize(title)
	desc = wordpunct_tokenize(desc)
	
	for topword in ['door','light','wall','shower','in','white','tile','vanity','outdoor','in','bathroom','led','paint','cabinet','sink','wood']:
		for word in words:
			if word == topword:
				freqword += 1

	search_numbers = len([int(s) for s in words if s.isdigit()])
	title_numbers = len([int(s) for s in title if s.isdigit()])
	desc_numbers = len([int(s) for s in desc if s.isdigit()])

	title_word_count = len(title)

	for word in words:
		t_word_count = 0
		'''
		for t_word in title:
			t_word_count += 1
			if stemmer.stem(word.lower()) == stemmer.stem(t_word.lower()):
				n_title += 1
				if t_word_count == 1:
					brand_match = 1
				if t_word_count == title_word_count:
					last_match = 1
				if t_word.isdigit():
					title_num_match = 1
				break
		for d_word in desc:
			if stemmer.stem(word.lower()) == stemmer.stem(d_word.lower()):
				n_desc += 1
				break
		'''
		for t_word in title:
			t_word_count +=1
			if word.lower() == t_word.lower():
				n_title += 1
				if t_word_count == 1:
					brand_match = 1
				if t_word_count == title_word_count:
					last_match = 1
				if t_word.isdigit():
					title_num_match = 1
				break

		for d_word in desc:
			if word.lower() == d_word.lower():
				n_desc += 1
				break
	
	return n_title, n_desc, len(words), search_chars, brand_match, last_match, title_numbers, desc_numbers, search_numbers, title_num_match, freqword

print("Get number of words and word matching title in train\n")
print(train.shape)
#train = train.head(0000)
#test = test.head(10000)
train_summary_stats = train.apply(lambda row: word_match(row['search_term'], row['product_title'], row['product_description']), axis=1)
#test_summary_stats = test.apply(lambda row: word_match(row['search_term'], row['product_title'], row['product_description']), axis=1)
X = np.array(pd.DataFrame.from_records(train_summary_stats.tolist()))
#X_test = pd.DataFrame.from_records(test_summary_stats.tolist())
y = train['relevance'].values

xgb_param = {'silent' : 1, 'eta': 0.2, 'objective': 'reg:linear'}

# do cross validation scoring
kf = KFold(train.shape[0], n_folds=4, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
for i, (tr_ix, val_ix) in enumerate(kf):
	dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix])
	dval = xgb.DMatrix(X[val_ix], y[val_ix])
	clf = xgb.train(xgb_param, dtrain, 40, evals=([dtrain,'train'], [dval,'val']))
	pred = clf.predict(dval)
	pred = np.clip(pred,1,3)
	scr[i] = np.sqrt(mean_squared_error(np.array(pred), y[val_ix]))
	print('CV score is:', scr[i])
print('Avg score is:', np.mean(scr))

