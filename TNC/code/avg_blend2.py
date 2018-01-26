import pandas as pd
import numpy as np

print("Loading OOB predictions...\n") 

model_output = ['tnc_sub.csv', #1.03057
				'tnc_vgg16_sub_2.csv', #1.02659
				'tnc_resnet_sub_2.csv', #1.00257
				'tnc_vgg19_sub.csv', #1.04188
				'tnc_vgs_sub.csv'] #1.03944

weight = [0.2,0.2,0.3,0.15,0.15]

for i, model_name in enumerate(model_output):
	model = pd.read_csv('../output/' + model_name)
	if i==0:
		model_avg = model
		model_avg.iloc[:,1:] = model.iloc[:,1:] * weight[i]
	else:
		model_avg.iloc[:,1:] += model.iloc[:,1:] * weight[i]

row_sums = np.sum(model_avg.iloc[:,1:], axis=1)
model_avg.iloc[:,1:] = np.array(model_avg.iloc[:,1:]) / row_sums[: , None]

print("Saving Results.")
model_pathname = '../output/tnc_resnet_vgg1619S_sub'
model_avg.to_csv(model_pathname + '.csv', index=False, float_format='%.7f')
