import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
	
def make_submission(clf, X_test, ids, encoder, name='rf_calibrated1.csv'):
	y_prob = clf.predict_proba(X_test)
	preds = pd.DataFrame(y_prob, index=ids, columns=encoder.classes_)
	preds.to_csv(name, index_label='id', float_format='%.4f')
	print("Wrote submission to file {}.".format(name))

def plot_calibration_curve(est, name, fig_index):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_val)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_val)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = log_loss(y_val, prob_pos)
        print("%s:" % name)
        print("\tLog Loss: %1.3f" % (clf_score))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_val, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


X, y, encoder, scaler = load_train_data('trainlg.csv')
X_test, ids = load_test_data('testlg.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.1)

# Plot calibration cuve for Gaussian Naive Bayes
plot_calibration_curve(RandomForestClassifier(), "Random Forest", 1)

# Train random forest classifier, calibrate on validation data and evaluate
# on test data
#kf = StratifiedShuffleSplit(y_train,n_iter=5, test_size=0.2, random_state=12)
# clf = RandomForestClassifier(n_estimators=500, max_features=32)
# sig_clf = CalibratedClassifierCV(clf, method="isotonic", cv=kf)

# sig_clf.fit(X_train, y_train)
# clf_probs = sig_clf.predict_proba(X_val)
# score = log_loss(y_val, clf_probs)
# print(score)

# make_submission(sig_clf, X_test, ids, encoder)
