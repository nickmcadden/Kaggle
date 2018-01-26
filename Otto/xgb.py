import xgboost as xgb
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold, StratifiedShuffleSplit
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import log_loss

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    return X, y, encoder

def load_test_data(path):
	df = pd.read_csv(path)
	X = df.values.copy()
	X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
	return X, ids
	
def make_submission(clf, X_test, ids, encoder, name='ert_calibrated3.csv'):
	y_prob = clf.predict_proba(X_test)
	preds = pd.DataFrame(y_prob, index=ids, columns=encoder.classes_)
	preds.to_csv(name, index_label='id', float_format='%.4f')
	print("Wrote submission to file {}.".format(name))

X, y, encoder = load_train_data('train.csv')
X_test, ids = load_test_data('test.csv')
num_classes = len(encoder.classes_)
num_features = X.shape[1]

rng = np.random.RandomState(2)
print("Otto: multiclass classification")
kf = StratifiedShuffleSplit(y,n_iter=3, test_size=0.33, random_state=12)
for train_index, test_index in kf:
	xgb_model = xgb.XGBClassifier(max_depth=12, 
									learning_rate=0.0057, 
									n_estimators=4000,
									objective="multi:softprob",
									nthread=-1,
									min_child_weight=5, 
									subsample=0.865, 
									colsample_bytree=0.55)
	xgb_model.fit(X[train_index],y[train_index])
	pred = xgb_model.predict_proba(X[test_index])
	score = log_loss(y[test_index], pred)
	print(score)
