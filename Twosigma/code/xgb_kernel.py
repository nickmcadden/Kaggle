import os
import sys
import operator
import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import pickle as pkl
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold, StratifiedKFold
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
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
	
def category_transformation(train_categoric, test_categoric, labels, type='std'):

	if type == 'freq':
		print("Encoding categories by freqency rank...")
		for c in train_categoric.columns:
			freqs = train_categoric[c].append(test_categoric[c]).value_counts()
			train_categoric[c] = pd.match(train_categoric[c].values, freqs[0:1000].index)
			test_categoric[c] = pd.match(test_categoric[c].values, freqs[0:1000].index)

	if type == 'tgtrate':
		print("Encoding categories by target rate...")
		for c in train_categoric.columns:
			train_categoric[c], test_categoric[c] = category_to_prob_weight(train_categoric, test_categoric, c, labels)

	return train_categoric, test_categoric

def add_median_price(key=None, suffix="", trn_df=None, tst_df=None):
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

train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json')

interest_level_map = {'high':0, 'medium':1, 'low':2}

y = train['interest_level'].apply(lambda x: interest_level_map[x])
ids = np.array(test['listing_id'])

print("Creating main features")

train["half_bathroom"] = ((train["bathrooms"] - np.floor(train["bathrooms"])) > 0).astype(int)
test["half_bathroom"] = ((test["bathrooms"] - np.floor(test["bathrooms"])) > 0).astype(int)

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

train["east"] = train["street_address"].apply(lambda x: x.find('East')>-1).astype(int)
test["east"] = test["street_address"].apply(lambda x: x.find('East')>-1).astype(int)

train["west"] = train["street_address"].apply(lambda x: x.find('West')>-1).astype(int)
test["west"] = test["street_address"].apply(lambda x: x.find('West')>-1).astype(int)

train["latlon"] = (train["latitude"]-train["longitude"]).astype('object')
test["latlon"] = (test["latitude"]-test["longitude"]).astype('object')

train["latlon"] = train["latlon"].apply(lambda x: 's' + str(x))
test["latlon"] = test["latlon"].apply(lambda x: 's' + str(x))

# convert the created column to datetime object so as to extract more features 
train["created"] = pd.to_datetime(train["created"])
train["days_since"] = train["created"].max() - train["created"]
train["days_since"] = (train["days_since"] / np.timedelta64(1, 'D')).astype(int)

created_rank = train["created"].rank()
listing_rank = train["listing_id"].rank()
train["freshness"] = created_rank - listing_rank

test["created"] = pd.to_datetime(test["created"])
test["days_since"] = test["created"].max() - test["created"]
test["days_since"] = (test["days_since"] / np.timedelta64(1, 'D')).astype(int)

created_rank = test["created"].rank()
listing_rank = test["listing_id"].rank()
test["freshness"] = created_rank - listing_rank

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

categorical = ["manager_id", "building_id"]

categorical_new = ["manager_id_count", "building_id_count"]

train_cat_temp = train[categorical]
test_cat_temp = test[categorical]

train[categorical], test[categorical] = category_transformation(train[categorical], test[categorical], y, 'tgtrate') 
train[categorical_new], test[categorical_new] = category_transformation(train_cat_temp, test_cat_temp, y, 'freq') 

categorical.extend(categorical_new)

print('Adding price to median price ratio data')
train, test = add_median_price(key=["bedrooms"], suffix="bed", trn_df=train, tst_df=test)

train['features'] = train["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test['features'] = test["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

tfidf = CountVectorizer(stop_words='english', max_features=200)
train_sparse_features = tfidf.fit_transform(train["features"])
test_sparse_features = tfidf.transform(test["features"])

original_features  = ["bathrooms", "bedrooms", "price", "listing_id", "latitude", "longitude"]
date_features = ["created_year", "created_month", "created_day", "created_hour", "days_since", "freshness"] # "created_weekday"
room_features = ["half_bathroom", "price_per_bed", "room_diff", "room_sum", "price_per_room", "beds_percent"] # "price_per_bath"
word_features = ["num_keywords", "num_description_words", "num_capital_letters", "num_address_words"] # "num_description_chars"
other_features = ["num_photos", "price_to_median_ratio_bed"] # "cool_door_num", "east", "west", "building_door_num"

for i in train.columns:
	print(i, train[i].dtype)

all_features = original_features + date_features + room_features + word_features + other_features + categorical# + img_data_cols_categorical + img_data_cols_date

train = train[all_features]
test = test[all_features]
X = sparse.hstack([train, train_sparse_features]).tocsr()
X_sub = sparse.hstack([test, test_sparse_features]).tocsr()
y = y.values

xgb_param = {'silent' : 1, 'eta': 0.1, 'max_depth':6, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'subsample': 0.7, 'num_class': 3, 'min_child_weight': 2, 'colsample_bytree': 0.7}

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=1)
scr = np.zeros([len(kf)])
oob_pred = np.zeros((X.shape[0], 3))
sub_pred = np.zeros((X_sub.shape[0], 3))
dtest = xgb.DMatrix(X_sub)
for i, (tr_ix, val_ix) in enumerate(kf):
	# get custom lookup for this fold
	print("CV fold: %d\n" %i)
	dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix])
	dval = xgb.DMatrix(X[val_ix], y[val_ix])
	clf = xgb.train(xgb_param, dtrain, 200, evals=([dtrain,'train'], [dval,'val']))
	pred = clf.predict(dval)
	oob_pred[val_ix] = np.array(pred)
	sub_pred += clf.predict(dtest)
	scr[i] = log_loss(y[val_ix], np.array(pred))
	print('CV score is: %f' % scr[i])

print('Avg score is:', np.mean(scr))
print oob_pred[1:10]

sub_pred = sub_pred / 5
oob_pred_filename = '../output/oob_pred_xgb_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_xgb_' + str(np.mean(scr))
pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
preds = pd.DataFrame({"listing_id": ids, "high": sub_pred[:,0], "medium": sub_pred[:,1], "low": sub_pred[:,2]})
preds = preds[["listing_id", "high", "medium", "low"]]
preds.to_csv('../output/xgb_' + str(np.mean(scr)) + '.csv', index=False)

'''
# Train on full data
dtrain = xgb.DMatrix(X,y)
dtest = xgb.DMatrix(X_sub)
clf = xgb.train(xgb_param, dtrain, 1500, evals=([dtrain,'train'], [dtrain,'train']))
pred = clf.predict(dtest)

print("Saving Results.")
preds = pd.DataFrame({"listing_id": ids, "high": pred[:,0], "medium": pred[:,1], "low": pred[:,2]})
preds = preds[["listing_id", "high", "medium", "low"]]
preds.to_csv('../output/xgb_full' + '.csv', index=False)
'''
