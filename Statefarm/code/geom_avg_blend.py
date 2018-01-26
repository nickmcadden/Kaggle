import pandas as pd
import numpy as np

print("Loading OOB predictions...\n") 

model_output = [
				'submission_vgg16_0.csv',
				'submission_vgg16_1.csv',
				'submission_vgg16_2.csv',
				'submission_vgg16_3.csv',
				'submission_vgg16_4.csv',
				'submission_vgg16_5.csv']

for i, model_name in enumerate(model_output):
	model = pd.read_csv('../output/' + model_name)
	if i==0:
		model_avg = model
	else:
		model_avg.iloc[:,1:] *= model.iloc[:,1:]
		
model_avg.iloc[:,1:] = np.power(model_avg.iloc[:,1:],1.0/(i+1.0))

print("Saving Results.")
model_pathname = '../output/blended_avg_'
model_avg.to_csv(model_pathname + '.csv', index=False, float_format='%.7f')
