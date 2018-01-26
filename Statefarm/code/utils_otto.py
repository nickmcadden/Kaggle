import numpy as np
from nolearn.lasagne import BatchIterator
from nolearn.lasagne.base import _sldict
from scipy.ndimage import rotate
from sklearn.utils import shuffle

def float32(k):
	return np.cast['float32'](k)


class CVTrainSplit(object):
    def __init__(self, cv):
        self.cv = cv

    def __call__(self, X, y, net=None):
        train_indices, valid_indices = next(iter(self.cv))
        X_train, y_train = _sldict(X, train_indices), y[train_indices]
        X_valid, y_valid = _sldict(X, valid_indices), y[valid_indices]
        return X_train, X_valid, y_train, y_valid


class RotateBatchIterator(BatchIterator):
    def __init__(self, max_angle=20, rotate_prob=1., *args, **kwargs):
        self.max_angle = max_angle
        self.rotate_prob = rotate_prob
        super(RotateBatchIterator, self).__init__(*args, **kwargs)

    def transform(self, X, y):
        if np.random.rand() > self.rotate_prob:
            return X, y

        angle = (np.random.rand() - 0.5) * 2 * self.max_angle
        X_new = np.zeros_like(X)
        for i, x in enumerate(X):
            X_new[i, 0] = rotate(x[0], angle, mode='nearest', reshape=False)
            #X_new, y = shuffle(X_new, y, random_state=0)
        return X_new, y


class EarlyStopping(object):
    def __init__(self, patience=3):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()


class AdjustVariable(object):
    def __init__(self, name, start=0.02, stop=0.005):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

