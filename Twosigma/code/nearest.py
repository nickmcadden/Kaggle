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
clusters = pd.unique(train["geo_cluster"])
#clusters = [85, 36]

tt = train.append(test)
tt.set_index(tt["listing_id"], inplace=True)

for c in clusters:
	print("cluster", c)
	tt_temp = tt[tt["geo_cluster"] == c]
	print(len(tt_temp))
	tt_temp["latitude"] *= 1000000
	tt_temp["longitude"] *= 1000000
	for i, row1 in enumerate(tt_temp.itertuples()):
		if i % 100 == 0:
			print(i)
		min_dist_list = [50000.0]*5
		min_dist = 50000.0
		dist = 0.0
		idx = row1.index
		for j, row2 in enumerate(tt_temp.itertuples()):
			if row1.listing_id == row2.listing_id:
				continue
			else:
				dist = np.sqrt(np.power(row1.latitude - row2.latitude, 2) + np.power(row1.longitude - row2.longitude, 2))
				if dist < min_dist:
					min_dist = dist
					min_dist_list.pop()
					min_dist_list.insert(0, min_dist)
		#print(min_dist, min_dist_list)
		tt.loc[row1.listing_id, 'nearest_dist'] = np.mean(min_dist_list)
	print(tt[tt["geo_cluster"] == c][["listing_id", "nearest_dist"]])
	
tt[["listing_id", "nearest_dist"]].to_csv("../input/nearest.csv", index=False)

