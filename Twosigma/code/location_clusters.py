import pandas as pd
import numpy as np
import csv
from sklearn.cluster import KMeans

train_file =  "../input/train.json"
test_file =  "../input/test.json"

print("Reading data files")
train_geo = pd.read_json(train_file)[["listing_id", "latitude", "longitude"]]
test_geo = pd.read_json(test_file)[["listing_id", "latitude", "longitude"]]

listings = train_geo.append(test_geo)

lat_min = np.min(listings["latitude"])
lat_width = np.max(listings["latitude"]) - np.min(listings["latitude"])

lon_min = np.min(listings["longitude"])
lon_width = np.max(listings["longitude"]) - np.min(listings["longitude"])

listings["latitude"] = (listings["latitude"] - lat_min)/ lat_width
listings["longitude"] = (listings["longitude"] - lon_min)/ lon_width

ids = listings["listing_id"]
listings.drop(["listing_id"], inplace=True, axis=1)

print("Clustering")
clst = KMeans(n_clusters=15, random_state=0, verbose=1).fit_predict(listings)

geo_clusters = pd.DataFrame({"listing_id": ids, "geo_cluster": clst}) 
geo_clusters.to_csv('../input/geo_clusters15.csv', index=False)
