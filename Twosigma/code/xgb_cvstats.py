import numpy as np
import pandas as pd
from scipy import sparse
import xgboost as xgb
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl
import argparse
from sklearn.cross_validation import KFold

train_df = pd.read_json("../input/train.json")
test_df = pd.read_json("../input/test.json")

image_date = pd.read_csv("../input/listing_image_time.csv")

# rename columns so you can join tables later on
image_date.columns = ["listing_id", "time_stamp"]

# reassign the only one timestamp from April, all others from Oct/Nov
image_date.loc[80240,"time_stamp"] = 1478129766

image_date["img_date"]                  = pd.to_datetime(image_date["time_stamp"], unit="s")
image_date["img_days_passed"]           = (image_date["img_date"].max() - image_date["img_date"]).astype("timedelta64[D]").astype(int)
image_date["img_date_month"]            = image_date["img_date"].dt.month
image_date["img_date_week"]             = image_date["img_date"].dt.week
image_date["img_date_day"]              = image_date["img_date"].dt.day
image_date["img_date_dayofweek"]        = image_date["img_date"].dt.dayofweek
image_date["img_date_dayofyear"]        = image_date["img_date"].dt.dayofyear
image_date["img_date_hour"]             = image_date["img_date"].dt.hour
image_date["img_date_monthBeginMidEnd"] = image_date["img_date_day"].apply(lambda x: 1 if x<10 else 2 if x<20 else 3)

train_df = pd.merge(train_df, image_date, on="listing_id", how="left")
test_df = pd.merge(test_df, image_date, on="listing_id", how="left")

ids = test_df["listing_id"].values

test_df["bathrooms"].loc[19671] = 1.5
test_df["bathrooms"].loc[22977] = 2.0
test_df["bathrooms"].loc[63719] = 2.0
train_df["price"] = train_df["price"].clip(upper=13000)

train_df["logprice"] = np.log(train_df["price"])
test_df["logprice"] = np.log(test_df["price"])

train_df['half_bathrooms'] = train_df["bathrooms"] - train_df["bathrooms"].apply(int)#.astype(int) # Half bathrooms? 1.5, 2.5, 3.5...
test_df['half_bathrooms'] = test_df["bathrooms"] - test_df["bathrooms"].apply(int)#.astype(int) # Half bathrooms? 1.5, 2.5, 3.5...

train_df["price_t"] =train_df["price"]/train_df["bedrooms"]
test_df["price_t"] = test_df["price"]/test_df["bedrooms"] 

train_df["room_sum"] = train_df["bedrooms"]+train_df["bathrooms"] 
test_df["room_sum"] = test_df["bedrooms"]+test_df["bathrooms"] 

train_df['price_per_room'] = train_df['price']/train_df['room_sum']
test_df['price_per_room'] = test_df['price']/test_df['room_sum']

train_df["num_photos"] = train_df["photos"].apply(len)
test_df["num_photos"] = test_df["photos"].apply(len)

train_df["num_features"] = train_df["features"].apply(len)
test_df["num_features"] = test_df["features"].apply(len)

train_df["num_description_words"] = train_df["description"].apply(lambda x: len(x.split(" ")))
test_df["num_description_words"] = test_df["description"].apply(lambda x: len(x.split(" ")))

train_df["created"] = pd.to_datetime(train_df["created"])
test_df["created"] = pd.to_datetime(test_df["created"])
train_df["created_year"] = train_df["created"].dt.year
test_df["created_year"] = test_df["created"].dt.year
train_df["created_month"] = train_df["created"].dt.month
test_df["created_month"] = test_df["created"].dt.month
train_df["created_day"] = train_df["created"].dt.day
test_df["created_day"] = test_df["created"].dt.day
train_df["created_hour"] = train_df["created"].dt.hour
test_df["created_hour"] = test_df["created"].dt.hour

train_df["created_weekday"] = train_df["created"].dt.weekday
test_df["created_weekday"] = test_df["created"].dt.weekday
train_df["created_week"] = train_df["created"].dt.week
test_df["created_week"] = test_df["created"].dt.week

train_df["pos"] = train_df.longitude.round(3).astype(str) + '_' + train_df.latitude.round(3).astype(str)
test_df["pos"] = test_df.longitude.round(3).astype(str) + '_' + test_df.latitude.round(3).astype(str)

vals = train_df['pos'].value_counts()
dvals = vals.to_dict()
train_df["density"] = train_df['pos'].apply(lambda x: dvals.get(x, vals.min()))
test_df["density"] = test_df['pos'].apply(lambda x: dvals.get(x, vals.min()))

features_to_use=["bathrooms", "bedrooms", "latitude", "longitude", "price", "price_t", "price_per_room", "logprice", "density", "half_bathrooms",
"num_photos", "num_features", "num_description_words","listing_id", "created_year", "created_month", "created_day", "created_hour", "created_week", "created_weekday",
"img_days_passed", "img_date_month", "img_date_week", "img_date_day", "img_date_dayofweek", "img_date_dayofyear", "img_date_hour", "img_date_monthBeginMidEnd"]

index=list(range(train_df.shape[0]))
random.shuffle(index)
a=[np.nan]*len(train_df)
b=[np.nan]*len(train_df)
c=[np.nan]*len(train_df)

for i in range(8):
    print("Manager stats:", i)
    building_level={}
    for j in train_df['manager_id'].values:
        building_level[j]=[0,0,0]
    
    test_index=index[int((i*train_df.shape[0])/8):int(((i+1)*train_df.shape[0])/8)]
    train_index=list(set(index).difference(test_index))
    
    for j in train_index:
        temp=train_df.iloc[j]
        if temp['interest_level']=='low':
            building_level[temp['manager_id']][0]+=temp["room_sum"]
        if temp['interest_level']=='medium':
            building_level[temp['manager_id']][1]+=temp["room_sum"]
        if temp['interest_level']=='high':
            building_level[temp['manager_id']][2]+=temp["room_sum"]

    for j in test_index:
        temp=train_df.iloc[j]
        if sum(building_level[temp['manager_id']])!=0:
            a[j]=building_level[temp['manager_id']][0]*1.0#/sum(building_level[temp['manager_id']])
            b[j]=building_level[temp['manager_id']][1]*1.0#/sum(building_level[temp['manager_id']])
            c[j]=building_level[temp['manager_id']][2]*1.0#/sum(building_level[temp['manager_id']])

train_df['manager_level_low']=a
train_df['manager_level_medium']=b
train_df['manager_level_high']=c

a=[]
b=[]
c=[]
building_level={}
for j in train_df['manager_id'].values:
    building_level[j]=[0,0,0]

for j in range(train_df.shape[0]):
    temp=train_df.iloc[j]
    if temp['interest_level']=='low':
        building_level[temp['manager_id']][0]+=temp["room_sum"]
    if temp['interest_level']=='medium':
        building_level[temp['manager_id']][1]+=temp["room_sum"]
    if temp['interest_level']=='high':
        building_level[temp['manager_id']][2]+=temp["room_sum"]

for i in test_df['manager_id'].values:
    if i not in building_level.keys():
        a.append(np.nan)
        b.append(np.nan)
        c.append(np.nan)
    else:
        a.append(building_level[i][0]*1.0)#/sum(building_level[i]))
        b.append(building_level[i][1]*1.0)#/sum(building_level[i]))
        c.append(building_level[i][2]*1.0)#/sum(building_level[i]))

test_df['manager_level_low']=a
test_df['manager_level_medium']=b
test_df['manager_level_high']=c

features_to_use.append('manager_level_low') 
features_to_use.append('manager_level_medium') 
features_to_use.append('manager_level_high')

categorical = ["street_address", "display_address", "manager_id", "building_id"]
for f in categorical:
        if train_df[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(train_df[f].values) + list(test_df[f].values))
            train_df[f] = lbl.transform(list(train_df[f].values))
            test_df[f] = lbl.transform(list(test_df[f].values))
            features_to_use.append(f)

train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

tfidf = CountVectorizer(stop_words='english', max_features=160)
tr_sparse = tfidf.fit_transform(train_df["features"])
te_sparse = tfidf.transform(test_df["features"])

train_X = sparse.hstack([train_df[features_to_use], tr_sparse]).tocsr()
test_X = sparse.hstack([test_df[features_to_use], te_sparse]).tocsr()

target_num_map = {'high':0, 'medium':1, 'low':2}
train_y = np.array(train_df['interest_level'].apply(lambda x: target_num_map[x]))

xgb_param = {'silent' : 1, 'eta': 0.03, 'max_depth':6, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'subsample': 0.6, 'num_class': 3, 'min_child_weight': 1, 'colsample_bytree': 0.6, 'seed': 100}

X, y, X_sub = train_X, train_y, test_X

# do cross validation scoring
kf = KFold(X.shape[0], n_folds=5, shuffle=True, random_state=3)
scr = np.zeros([len(kf)])
oob_pred = np.zeros((X.shape[0], 3))
sub_pred = np.zeros((X_sub.shape[0], 3))
dtest = xgb.DMatrix(X_sub)
for i, (tr_ix, val_ix) in enumerate(kf):
	# get custom lookup for this fold
	print("CV fold: %d\n" %i)
	dtrain = xgb.DMatrix(X[tr_ix], y[tr_ix])
	dval = xgb.DMatrix(X[val_ix], y[val_ix])
	clf = xgb.train(xgb_param, dtrain, 1000, evals=([dtrain,'train'], [dval,'val']))
	pred = clf.predict(dval)
	oob_pred[val_ix] = np.array(pred)
	sub_pred += clf.predict(dtest)
	scr[i] = log_loss(y[val_ix], np.array(pred))
	print('CV score is: %f' % scr[i])

print('Avg score is:', np.mean(scr))

sub_pred = sub_pred / 5
oob_pred_filename = '../output/oob_pred_xgb_cvs_' + str(np.mean(scr))
sub_pred_filename = '../output/sub_pred_xgb_cvs_' + str(np.mean(scr))
pkl.dump(oob_pred, open(oob_pred_filename + '.p', 'wb'))
pkl.dump(sub_pred, open(sub_pred_filename + '.p', 'wb'))
preds = pd.DataFrame({"listing_id": ids, "high": sub_pred[:,0], "medium": sub_pred[:,1], "low": sub_pred[:,2]})
preds = preds[["listing_id", "high", "medium", "low"]]
preds.to_csv('../output/xgb_cvstats' + str(np.mean(scr)) + '.csv', index=False)

