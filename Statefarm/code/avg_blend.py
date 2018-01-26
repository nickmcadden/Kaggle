import pandas as pd
import numpy as np

print("Loading OOB predictions...\n") 

model_output = [
				'submission_vgg16_0.csv',
				'submission_vgg16_1.csv',
				'submission_vgg16_2.csv',
				'submission_vgg16_3.csv',
				'submission_vgg16_4.csv',
				'submission_vgg16_5.csv',
				'submission_vgg16_6.csv',
				'submission_vgg16_7.csv']

weight = [0.125]*8

for i, model_name in enumerate(model_output):
	model = pd.read_csv('../output/J23/' + model_name)
	if i==0:
		model_avg = model
		model_avg.iloc[:,1:] = model.iloc[:,1:] * weight[i]
	else:
		model_avg.iloc[:,1:] += model.iloc[:,1:] * weight[i]

#model_avg.iloc[:,1:] /= (i+1)

print("Saving Results.")
model_pathname = '../output/J23_8'
model_avg.to_csv(model_pathname + '.csv', index=False, float_format='%.7f')
