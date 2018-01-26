import pandas as pd
import numpy as np
from collections import Counter
import sys

print("Reading data\n")
train = pd.read_json('../input/train.json', encoding="ISO-8859-1").reindex()
test = pd.read_json('../input/test.json', encoding="ISO-8859-1").reindex()

print("merging with geocoded region data")

geo_clusters = pd.read_csv('../input/geo_clusters.csv')

train = pd.merge(train, geo_clusters, on="listing_id")
test = pd.merge(test, geo_clusters, on="listing_id")

train.drop(["interest_level"], axis=1, inplace=True)

print(train.shape, test.shape)
clusters = pd.unique(train["building_id"])

tt = train.append(test)
tt.set_index(tt["listing_id"], inplace=True)

tt["created"] = pd.to_datetime(tt["created"])
tt["listing_day"] = tt["created"] - tt["created"].min()
tt["listing_day"] = (tt["listing_day"] / np.timedelta64(1, 'D')).astype(int)

for c in clusters:
	print("cluster", c)
	tt_temp = tt[tt["building_id"] == c]
	beds = list(pd.unique(tt_temp["bedrooms"]))
	for b in beds:
		#print("bedrooms", b)
		tt_temp_beds = tt_temp[tt_temp["bedrooms"]==b]
		tt_temp_beds = tt_temp_beds.sort_values(by=["listing_day"])
		#print(len(tt_temp_beds))
		last_listing_day = 0
		for i, row1 in enumerate(tt_temp_beds.itertuples()):
			#if i % 100 == 0:
				#print(i)
			tt.loc[row1.listing_id, 'days_since_last'] = row1.listing_day - last_listing_day
			last_listing_day = row1.listing_day
	print(tt[tt["building_id"] == c][["listing_id", "days_since_last"]])

tt[["listing_id", "days_since_last"]].to_csv("../input/recency.csv", index=False)
