import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.cross_validation import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import log_loss

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
	df = pd.read_csv(path)
	X = df.values.copy()
	X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
	X = scaler.transform(X)
	return X, ids
	
def make_submission(clf, X_test, ids, encoder, name='ert_calibrated3.csv'):
	y_prob = clf.predict_proba(X_test)
	preds = pd.DataFrame(y_prob, index=ids, columns=encoder.classes_)
	preds.to_csv(name, index_label='id', float_format='%.4f')
	print("Wrote submission to file {}.".format(name))

# Get the data
np.random.seed(2)
	
X, y, encoder, scaler = load_train_data('trainlg.csv')
X_test, ids = load_test_data('testlg.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.1)

kf = StratifiedShuffleSplit(y_train,n_iter=5, test_size=0.2, random_state=12)
clf = ExtraTreesClassifier(n_estimators=4500, max_features=32, n_jobs=16, verbose=1)
iso_clf = CalibratedClassifierCV(clf, method="isotonic", cv=kf)
iso_clf.fit(X, y)
#iso_clf.fit(X_train, y_train)
#clf_probs = iso_clf.predict_proba(X_val)
#score = log_loss(y_val, clf_probs)
#print(score)

make_submission(iso_clf, X_test, ids, encoder)
