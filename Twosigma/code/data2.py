import pandas as pd
import numpy as np
import string
import re
import time
import gc as gc
from scipy import sparse
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.snowball import EnglishStemmer
from nltk.util import ngrams
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def prob_weight_lookup(code, lookup, labels):
	try:
		if lookup.ix[code, 'freq'] >= 3: 
			return lookup.ix[code, 'tgt_rate_adj']
		else:
			return np.mean(labels)
	except:
		return np.mean(labels)

def category_to_prob_weight(train, test, col, labels):

	traincol, testcol, labels = pd.Series(train[col]), pd.Series(test[col]), pd.Series(labels)
	kf = StratifiedKFold(labels, n_folds=5, shuffle=True, random_state=1)
	traincolprob = np.zeros(traincol.shape[0])
	print(col)
	for kfold, (tr_ix, val_ix) in enumerate(kf):
		print('train data: fold:', kfold)
		train_tr = traincol.iloc[tr_ix]
		train_val = traincol.iloc[val_ix]
		freqs = pd.DataFrame(train_tr.value_counts())
		corr = pd.concat([train_tr, labels.iloc[tr_ix]], axis=1)
		corr = pd.DataFrame(corr.groupby(col).mean())
		lookup = pd.merge(corr, freqs, how='outer', left_index=True, right_index=True)
		lookup.columns = (['target','freq'])
		# Bayesian aspect - tend towards mean target % for levels with low freq count
		lookup['tgt_rate_adj'] = ((lookup['freq'] - 1) * lookup['target'] + np.mean(labels)) / lookup['freq']
		traincolprob[val_ix] = train_val.apply(lambda row: prob_weight_lookup(row, lookup, labels))
	print('test data')
	testcolprob = testcol.apply(lambda row: prob_weight_lookup(row, lookup, labels))

	return traincolprob, testcolprob

def bayesian_average(train, test, col, y):

    lambda_val = None
    k = 5.0
    f = 1.0
    r_k = 0.01
    g = 1.0

    print(col)

    def calculate_average(sub1, sub2, prior):
        s = pd.DataFrame(data = {
                                 col: sub1.groupby(col, as_index = False).count()[col],                              
                                 'sumy': sub1.groupby(col, as_index = False).sum()['y'],
                                 'avgY': sub1.groupby(col, as_index = False).mean()['y'],
                                 'cnt': sub1.groupby(col, as_index = False).count()['y']
                                 })

        tmp = sub2.merge(s.reset_index(), how = 'left', on = col)
        del tmp['index']
        tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
        tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0

        def compute_beta(row):
            cnt = row['cnt'] if row['cnt'] < 200 else float('inf')
            return 1.0 / (g + np.exp((cnt - k) / f))

        if lambda_val is not None:
            tmp['beta'] = lambda_val
        else:
            tmp['beta'] = tmp.apply(compute_beta, axis = 1)

        tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * prior, axis = 1)

        tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = prior
        tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = prior
        tmp['random'] = np.random.uniform(size = len(tmp))
        tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] * (1 + (row['random'] - 0.5) * r_k), axis = 1)

        return tmp['adj_avg'].ravel()

    #cv for training set
    prior = y.mean()

    k_fold = StratifiedKFold(y, n_folds=5, shuffle=True, random_state=1)
    traincolprob = np.zeros(len(train))

    for i, (train_index, cv_index) in enumerate(k_fold):
        print("train cv")
        print(i)
        sub = pd.DataFrame({col: train[col], 'y': y})
        sub1 = sub.iloc[train_index]
        sub2 = sub.iloc[cv_index]

        traincolprob[cv_index] = calculate_average(sub1, sub2, prior)

    print('test')
    #for test set
    sub1 = pd.DataFrame({col: train[col], 'y': y})
    sub2 = pd.DataFrame({col: test[col]})

    testcolprob = calculate_average(sub1, sub2, prior)
    print('done test')
    return traincolprob, testcolprob

def category_transformation(train_categoric, test_categoric, labels, type='std'):

	if type == 'freq':
		print("Encoding categories by freqency rank...")
		for c in train_categoric.columns:
			freqs = train_categoric[c].append(test_categoric[c]).value_counts()
			train_categoric[c] = pd.match(train_categoric[c].values, freqs[0:1000].index)
			test_categoric[c] = pd.match(test_categoric[c].values, freqs[0:1000].index)

	if type == 'std':
		print("Encoding categories by sklearn label encoder...")
		for c in train_categoric.columns:
			lbl = LabelEncoder()
			lbl.fit(list(train_categoric.ix[:,c]) + list(test_categoric.ix[:,c]))
			train_categoric.ix[:,c] = lbl.transform(train_categoric.ix[:,c])
			test_categoric.ix[:,c] = lbl.transform(test_categoric.ix[:,c])

	if type == 'tgtrate':
		print("Encoding categories by target rate...")
		for c in train_categoric.columns:
			train_categoric[c], test_categoric[c] = category_to_prob_weight(train_categoric, test_categoric, c, labels)

	if type == 'bayesian':
		print("Encoding categories by bayesian prob...")
		for c in train_categoric.columns:
			train_categoric[c], test_categoric[c] = bayesian_average(train_categoric, test_categoric, c, labels)

	if type == 'rank':
		print("Encoding categories by rank transformation...")
		for c in train_categoric.columns:
			rank = pd.concat([train_categoric[c],labels], axis=1).groupby(c).mean().sort_values(by='interest_level', ascending=False)
			train_categoric[c] = pd.match(train_categoric[c].values, rank[0:20000].index)
			test_categoric[c] = pd.match(test_categoric[c].values, rank[0:20000].index)

	if type == 'onehot':
		print("One hot... ")
		for c in train_categoric.columns:
			uniques = np.unique(train_categoric[c])
			if len(uniques) > 100:
				train_categoric.drop(c, axis=1, inplace=True)
				test_categoric.drop(c, axis=1, inplace=True)
		x_cat_train = train_categoric.T.to_dict().values()
		x_cat_test = test_categoric.T.to_dict().values()

		# vectorize
		vectorizer = DV(sparse = False)
		train_categoric = pd.DataFrame(vectorizer.fit_transform(x_cat_train))
		test_categoric = pd.DataFrame(vectorizer.transform(x_cat_test))

	return train_categoric, test_categoric

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

def get_term_score_freq(words, lookup):
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

def word_match(product_uid, words, title, desc):
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

	return product_uid, n1gram_title, n1gram_desc, n2gram_title, n2gram_desc, n3gram_title, n3gram_desc, n1keypos_match, n2keypos_match, n3keypos_match, len(words), len(title), len(desc), search_chars, brand_match, title_numbers, desc_numbers, search_numbers, title_num_match, total_match, joining_word

def regex_first_number(address):
	firstnum =  re.findall(r'^\D*(\d+)', address)
	if len(firstnum):
		return int(firstnum[0])
	else:
		return 0


def add_median_price(key=None, suffix="", trn_df=None, tst_df=None):
    """
    Compute median prices for renthop dataset.
    The function adds 2 columns to the pandas DataFrames : the median prices and a ratio
    between nthe actual price of the rent and the median
    
    :param key: list of columns on which to groupby and compute median prices
    :param suffix: string used to suffix the newly created columns/features
    :param trn_df: training dataset as a pandas DataFrame
    :param tst_df: test dataset as a pandas DataFrame
    :return: updated train and test DataFrames

    :Example
    
    train, test = add_median_price(key=['bedrooms', 'bathrooms'], 
                                   suffix='rooms', 
                                   trn_df=train, 
                                   tst_df=test)

    """
    # Set features to be used
    median_features = key[:]
    median_features.append('price')
    # Concat train and test to find median prices over whole dataset
    median_prices = pd.concat([trn_df[median_features], tst_df[median_features]], axis=0)
    # Group data by key to compute median prices
    medians_by_key = median_prices.groupby(by=key)['price'].median().reset_index()
    # Rename median column with provided suffix
    medians_by_key.rename(columns={'price': 'median_price_' + suffix}, inplace=True)
    # Update data frames
    trn_df = trn_df.merge(medians_by_key, on=key, how="left")
    tst_df = tst_df.merge(medians_by_key, on=key, how="left")
    trn_df['price_to_median_ratio_' + suffix] = trn_df['price'] / trn_df['median_price_' + suffix]
    tst_df['price_to_median_ratio_' + suffix] = tst_df['price'] / tst_df['median_price_' + suffix]

    return trn_df, tst_df

def load(m_params):
	getcached = m_params['getcached']
	codetest = m_params['codetest']

	print("Reading data\n")

	train = pd.read_json('../input/train.json')
	test = pd.read_json('../input/test.json')

	print("merging with geocoded region data")

	geo_clusters = pd.read_csv('../input/geo_clusters.csv')

	train = pd.merge(train, geo_clusters, on="listing_id")
	test = pd.merge(test, geo_clusters, on="listing_id")

	'''	
	print("merging with sentiment data")

	train_sentiment = pd.read_csv('../input/train_sentiment.csv')
	test_sentiment = pd.read_csv('../input/test_sentiment.csv')

	train = pd.merge(train, train_sentiment, on="listing_id")
	test = pd.merge(test, test_sentiment, on="listing_id")

	sentiment = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust","negative","positive"]
	'''

	interest_level_map = {'high':0, 'medium':1, 'low':2}

	y = train['interest_level'].apply(lambda x: interest_level_map[x])
	ids = np.array(test['listing_id'])

	print("Creating main features")

	train["half_bathroom"] = ((train["bathrooms"] - np.floor(train["bathrooms"])) > 0).astype(int)
	test["half_bathroom"] = ((test["bathrooms"] - np.floor(test["bathrooms"])) > 0).astype(int)

	train["bedrooms"] = np.clip(train["bedrooms"],1,6)
	train["bathrooms"] = np.clip(train["bathrooms"],1,5)
	test["bedrooms"] = np.clip(test["bedrooms"],1,6)
	test["bathrooms"] = np.clip(test["bathrooms"],1,5)

	train["price_per_bed"] = train["price"]/(train["bedrooms"])
	train["price_per_bath"] = train["price"]/(train["bathrooms"]) 
	train["room_diff"] = train["bedrooms"]-train["bathrooms"]
	train["room_sum"] = train["bedrooms"]+train["bathrooms"]
	train["price_per_room"] = train["price"]/(train["room_sum"])
	train["beds_percent"] = train["bedrooms"]/(train["room_sum"])

	test["price_per_bed"] = test["price"]/(test["bedrooms"])
	test["price_per_bath"] = test["price"]/(test["bathrooms"]) 
	test["room_diff"] = test["bedrooms"]-test["bathrooms"]
	test["room_sum"] = test["bedrooms"]+test["bathrooms"]
	test["price_per_room"] = test["price"]/(test["room_sum"])
	test["beds_percent"] = test["bedrooms"]/(test["room_sum"])

	# count of photos
	train["num_photos"] = train["photos"].apply(len)
	test["num_photos"] = test["photos"].apply(len)

	# count of "features"
	train["num_keywords"] = train["features"].apply(len)
	test["num_keywords"] = test["features"].apply(len)

	# count of words present in description column #
	train["num_description_words"] = train["description"].apply(lambda x: len(x.split(" ")))
	test["num_description_words"] = test["description"].apply(lambda x: len(x.split(" ")))

	# count of words present in description column #
	train["num_capital_letters"] = train["description"].apply(lambda x: sum(1 for c in x if c.isupper()))
	test["num_capital_letters"] = test["description"].apply(lambda x: sum(1 for c in x if c.isupper()))

	# count of words present in description column #
	train["num_description_chars"] = train["description"].apply(lambda x: len(x))
	test["num_description_chars"] = test["description"].apply(lambda x: len(x))

	# is the building in an avenue, street etc #
	train["num_address_words"] = train["display_address"].apply(lambda x: len(x.split(" ")))
	test["num_address_words"] = test["display_address"].apply(lambda x: len(x.split(" ")))

	# is the building in an avenue, street etc #
	train["num_address_chars"] = train["display_address"].apply(lambda x: len(x))
	test["num_address_chars"] = test["display_address"].apply(lambda x: len(x))

	# is the building in an avenue, street etc #
	train["building_door_num"] = train["street_address"].apply(lambda x: regex_first_number(x))
	test["building_door_num"] = test["street_address"].apply(lambda x: regex_first_number(x))

	train["cool_door_num"] = (train["building_door_num"] < 100).astype(int)
	test["cool_door_num"] = (test["building_door_num"] < 100).astype(int)

	train["east"] = train["street_address"].apply(lambda x: x.find('East')>-1).astype(int)
	test["east"] = test["street_address"].apply(lambda x: x.find('East')>-1).astype(int)

	train["west"] = train["street_address"].apply(lambda x: x.find('West')>-1).astype(int)
	test["west"] = test["street_address"].apply(lambda x: x.find('West')>-1).astype(int)
	
	train["latlon"] = (train["latitude"]-train["longitude"]).astype('object')
	test["latlon"] = (test["latitude"]-test["longitude"]).astype('object')
	
	train["latlon"] = train["latlon"].apply(lambda x: 's' + str(x))
	test["latlon"] = test["latlon"].apply(lambda x: 's' + str(x))
	
	train["geo_clust_bed"] = train["geo_cluster"] * 10 + train["bedrooms"]
	test["geo_clust_bed"] = test["geo_cluster"] * 10 + test["bedrooms"]

	# convert the created column to datetime object so as to extract more features 
	train["created"] = pd.to_datetime(train["created"])
	train["days_since"] = train["created"].max() - train["created"]
	train["days_since"] = (train["days_since"] / np.timedelta64(1, 'D')).astype(int)

	test["created"] = pd.to_datetime(test["created"])
	test["days_since"] = test["created"].max() - test["created"]
	test["days_since"] = (test["days_since"] / np.timedelta64(1, 'D')).astype(int)

	# extract some features like year, month, day, hour from date columns
	train["created_year"] = train["created"].dt.year
	test["created_year"] = test["created"].dt.year
	train["created_month"] = train["created"].dt.month
	test["created_month"] = test["created"].dt.month
	train["created_day"] = train["created"].dt.day
	test["created_day"] = test["created"].dt.day
	train["created_hour"] = train["created"].dt.hour
	test["created_hour"] = test["created"].dt.hour
	train['created_weekday'] = train['created'].dt.weekday
	test['created_weekday'] = test['created'].dt.weekday

	print('Categorical variable transformation')

	categorical = ["manager_id", "building_id", "geo_clust_bed"]

	categorical_new = ["manager_id_count", "building_id_count", "geo_clust_bed_count"]

	train_cat_temp = train[categorical]
	test_cat_temp = test[categorical]

	'''
	def find_only_one_of(feature_name):
		temp = pd.concat([train[feature_name].reset_index(), test[feature_name].reset_index()])
		temp = temp.groupby(feature_name, as_index = False).count()
		return temp[temp['index'] == 1]

	for f in categorical:
		only_one = find_only_one_of(f)
		train.loc[train[f].isin(only_one[f].ravel()), f] = "-1"
		test.loc[test[f].isin(only_one[f].ravel()), f] = "-1"
	'''

	train[categorical], test[categorical] = category_transformation(train[categorical], test[categorical], y, 'tgtrate') 

	train[categorical_new], test[categorical_new] = category_transformation(train_cat_temp, test_cat_temp, y, 'freq') 

	categorical.extend(categorical_new)

	print('Adding price to median price ratio data')
	train, test = add_median_price(key=["bedrooms"], suffix="bed", trn_df=train, tst_df=test)

	train['features'] = train["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
	test['features'] = test["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
	#print(train["features"].head())
	tfidf = CountVectorizer(stop_words='english', max_features=200)
	train_sparse_features = tfidf.fit_transform(train["features"])
	test_sparse_features = tfidf.transform(test["features"])

	original_features  = ["bathrooms", "bedrooms", "price", "listing_id", "latitude", "longitude"]
	date_features = ["created_year", "created_month", "created_day", "created_hour", "days_since"] # "created_weekday"
	room_features = ["half_bathroom", "price_per_bed", "room_diff", "room_sum", "price_per_room", "beds_percent"] # "price_per_bath"
	word_features = ["num_keywords", "num_description_words", "num_capital_letters", "num_address_words"] # "num_description_chars"
	other_features = ["num_photos", "price_to_median_ratio_bed"] # "cool_door_num", "east", "west", "building_door_num"

	for i in train.columns:
		print(i, train[i].dtype)

	all_features = original_features + date_features + room_features + word_features + other_features + categorical# + img_data_cols_categorical + img_data_cols_date

	train = train[all_features]
	test = test[all_features]
	
	train.fillna(0, inplace=True)
	test.fillna(0, inplace=True)
	
	train = sparse.hstack([train, train_sparse_features]).tocsr()
	test = sparse.hstack([test, test_sparse_features]).tocsr()

	print(train.shape, test.shape)

	'''
	train_summary_stats = train.apply(lambda row: word_match(row['product_uid'], row['search_term'], row['product_title'], row['product_description']), axis=1)
	test_summary_stats = test.apply(lambda row: word_match(row['product_uid'], row['search_term'], row['product_title'], row['product_description']), axis=1)
	X = pd.DataFrame.from_records(train_summary_stats.tolist())
	X_sub = pd.DataFrame.from_records(test_summary_stats.tolist())

	kf = KFold(train.shape[0], n_folds=5, shuffle=True, random_state=1)
	X_append = np.zeros((X.shape[0], 2))
	for i, (tr_ix, val_ix) in enumerate(kf):
		# get custom lookup for this fold
		print("CV fold: %d\n" %i)
		print("Creating dictionary lookup")
		lookup = homedepot_dict_lookup(train.values[tr_ix])
		print("Creating fold specific features\n")
		term_score_freq = train.apply(lambda row: get_term_score_freq(row['search_term'], lookup), axis=1)
		X_append[val_ix] = pd.DataFrame.from_records(term_score_freq.tolist()).values[val_ix]

	X_append = pd.DataFrame(X_append)
	print(X_append.shape)

	X = pd.concat([X, X_append], axis=1)

	lookup = homedepot_dict_lookup(train.values)
	test_term_score_freq = test.apply(lambda row: get_term_score_freq(row['search_term'], lookup), axis=1)
	X_append = pd.DataFrame.from_records(test_term_score_freq.tolist())
	X_sub = pd.concat([X_sub, X_append], axis=1)
	'''

	return train, y.values, test, ids
 