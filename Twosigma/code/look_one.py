import pandas as pd
import numpy as np
from collections import Counter
import sys, pickle as pkl
from scipy.stats import pearsonr

print("Reading data\n")
#train = pd.read_json('../input/train.json', encoding="ISO-8859-1").reindex()
#test = pd.read_json('../input/test.json', encoding="ISO-8859-1").reindex()

oob_models = 	['oob_pred_etcentropy_0.551557316458.p',
				'oob_nnet_0.574446898612.p',
				'oob_pred_lr_0.574865487204.p',
				'oob_pred_gbcentropy_0.550568716867.p']

ref_model = pkl.load(open('../output/oob_pred_xgb_0.524605390397.p', 'rb'))

for oob_model_name in oob_models:
	oob_model = pkl.load(open('../output/' + oob_model_name,'rb'))
	print(oob_model_name)
	print(np.round(oob_model,2)[:10])
	print(pearsonr(oob_model[:, 2], ref_model[:, 2]))


print(created_rank-listing_rank)
