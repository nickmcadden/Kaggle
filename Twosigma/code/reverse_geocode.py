import pandas as pd
import csv
import reverse_geocoder as rg

train_file =  "../input/train.json"
test_file =  "../input/test.json"

train_geo = pd.read_json(train_file)[["listing_id", "latitude", "longitude"]]
print(len(train_geo))
test_geo = pd.read_json(test_file)[["listing_id", "latitude", "longitude"]]
print(len(test_geo))

listing_coords = train_geo.append(test_geo)

lat_lon = []
listings = []

for i, j in listing_coords.iterrows():
    lat_lon.append((j["latitude"], j["longitude"]))
    listings.append(int(j["listing_id"]))

results = rg.search(lat_lon)

nbd = [[listings[i], results[i]['name']] for i in range(0, len(results))] #getting ready to write to csv 
print(len(nbd))

with open("../input/neighborhoods.csv", "wb") as f:
    writer = csv.writer(f, delimiter = ",")
    writer.writerows(nbd)
