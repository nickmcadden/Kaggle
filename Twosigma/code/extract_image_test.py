import pandas as pd
import numpy as np
from PIL import Image, ExifTags
import os

tag_df = pd.DataFrame()
rowcount=0

for n, i in enumerate(os.walk('../input/images_sample/')):
	if n % 1000 == 0:
		print(n, rowcount)
	img_path = i[0]
	idx = i[0].split("/")[3]
	for j in range(len(i[2])):
		print(i[2][j])
		if i[2][j] != '.DS_Store':
			img = Image.open(img_path + '/' + i[2][j])
			try:
				meta = img._getexif()
				if meta is not None:
					rowcount += 1
					exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
					for k,v in exif.items():
						if k != 'MakerNote':
							try:
								tag_df.loc[idx, k] = v
							except:
								tag_df.loc[idx, k] = v[0]
							print(k,v)
			except:
				continue

print(tag_df)
tag_df.to_csv('../input/image_meta_data.csv', index=True)

'''
img = Image.open("/path/to/file.jpg")
exif = { ExifTags.TAGS[k]: v for k, v in img._getexif().items() if k in ExifTags.TAGS }
'''
