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
	
def make_submission(clf, X_test, ids, encoder, name='xgb_scikit_big.csv'):
	y_prob = clf.predict_proba(X_test)
	preds = pd.DataFrame(y_prob, index=ids, columns=encoder.classes_)
	preds.to_csv(name, index_label='id', float_format='%.4f')
	print("Wrote submission to file {}.".format(name))

X, y, encoder = load_train_data('train.csv')
X_test, ids = load_test_data('test.csv')
num_classes = len(encoder.classes_)
num_features = X.shape[1]

rng = np.random.RandomState(2)
print("Otto: multiclass classification BIG")
xgb_model = xgb.XGBClassifier(max_depth=12, 
							learning_rate=0.00197, 
							n_estimators=12000,
							objective="multi:softprob",
							nthread=-1,
							min_child_weight=5, 
							subsample=0.9, 
							colsample_bytree=0.6)
xgb_model.fit(X,y)
make_submission(xgb_model, X_test, ids, encoder)
