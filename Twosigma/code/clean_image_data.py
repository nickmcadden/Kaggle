import pandas as pd
import numpy as np
from PIL import Image, ExifTags
import os
import string

print("Reading data\n")
img_df = pd.read_csv('../input/image_meta_data.csv', encoding="utf-8")

cols_to_keep = ["listing_id"]
for i in img_df.columns:
	unique_values = len(np.unique(img_df[i]))
	null_values = np.sum(pd.isnull(img_df[i]))
	if img_df[i].dtype == 'O' and null_values < 7000:
		print(i, img_df[i].dtype)
		print('unique', unique_values)
		print('null', null_values)
		cols_to_keep.append(i)

print(cols_to_keep)

img_df["DateTimeDigitized"].fillna('2017/01/01', inplace=True)
img_df["DateTimeDigitized"] = img_df["DateTimeDigitized"].apply(lambda x: string.replace(x[0:10],':','/'))
img_df["DateTimeDigitized"] = img_df["DateTimeDigitized"].apply(lambda x: string.replace(x,'0000/00/00', '2017/01/01'))
img_df["DateTimeDigitized"] = img_df["DateTimeDigitized"].apply(lambda x: string.replace(x,'4/45/09', '2017/01/01'))
img_df["DateTimeDigitized"] = pd.to_datetime(img_df["DateTimeDigitized"])

img_df["DateTimeOriginal"].fillna('2017/01/01', inplace=True)
img_df["DateTimeOriginal"] = img_df["DateTimeOriginal"].apply(lambda x: string.replace(x[0:10],':','/'))
img_df["DateTimeOriginal"] = img_df["DateTimeOriginal"].apply(lambda x: string.replace(x,'0000/00/00', '2017/01/01'))
img_df["DateTimeOriginal"] = img_df["DateTimeOriginal"].apply(lambda x: string.replace(x,'4/45/09', '2017/01/01'))
img_df["DateTimeOriginal"] = pd.to_datetime(img_df["DateTimeOriginal"])

img_df["DateTime"].fillna('2017/01/01', inplace=True)
img_df["DateTime"] = img_df["DateTime"].apply(lambda x: string.replace(x[0:10],':','/'))
img_df["DateTime"] = img_df["DateTime"].apply(lambda x: string.replace(x[0:10],'-','/'))
img_df["DateTime"] = img_df["DateTime"].apply(lambda x: string.replace(x,'0000/00/00', '2017/01/01'))
img_df["DateTime"] = pd.to_datetime(img_df["DateTime"])


for i in img_df["DateTimeOriginal"]:
	try:
		pd.to_datetime(i)
		#print(i, pd.to_datetime(i))
	except:
		print i

img_df = img_df[cols_to_keep]
img_df.to_csv('../input/image_meta_data_cleaned.csv', index=False, encoding='utf-8')
