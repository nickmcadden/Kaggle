# Model definition for VGG-16, 16-layer model from the paper:
# "Very Deep Convolutional Networks for Large-Scale Image Recognition"
# Original source: https://gist.github.com/ksimonyan/211839e770f7b538e2d8

# More pretrained models are available from
# https://github.com/Lasagne/Recipes/blob/master/modelzoo/
import pickle
import gc
import numpy as np
import lasagne, sklearn, theano
import theano.tensor as T
from collections import OrderedDict
from sklearn.cross_validation import train_test_split, KFold, LabelKFold
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, DropoutLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.nonlinearities import softmax
from lasagne.utils import floatX
from scipy.ndimage import rotate
from keras.utils.generic_utils import Progbar
from data_pretrained import load_train_data, load_test_data, make_submission

gc.collect()
np.random.seed(1)
IMG_SHAPE = 224, 224

def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1, flip_filters=False)
    net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1, flip_filters=False)
    net['pool1'] = PoolLayer(net['conv1_2'], 2)
    net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1, flip_filters=False)
    net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1, flip_filters=False)
    net['pool2'] = PoolLayer(net['conv2_2'], 2)
    net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1, flip_filters=False)
    net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1, flip_filters=False)
    net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1, flip_filters=False)
    net['pool3'] = PoolLayer(net['conv3_4'], 2)
    net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1, flip_filters=False)
    net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1, flip_filters=False)
    net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1, flip_filters=False)
    net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1, flip_filters=False)
    net['pool4'] = PoolLayer(net['conv4_4'], 2)
    net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1, flip_filters=False)
    net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1, flip_filters=False)
    net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1, flip_filters=False)
    net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1, flip_filters=False)
    net['pool5'] = PoolLayer(net['conv5_4'], 2)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.6)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.6)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc8'], softmax)

    return net
# Load model weights and metadata
d = pickle.load(open('../input/pretrained/vgg19.pkl'))

# Build the network and fill with pretrained weights
net = build_model()

# Define loss function and metrics, and get an updates dictionary
X_sym = T.tensor4()
y_sym = T.ivector()

# We'll connect our output classifier to the last fully connected layer of the network
net['new_output'] = DenseLayer(net['drop7'], num_units=8, nonlinearity=softmax, W=lasagne.init.Normal(0.01)) 

prediction = lasagne.layers.get_output(net['new_output'], X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym), dtype=theano.config.floatX)

learning_rate = theano.shared(np.array(0.0001, dtype=theano.config.floatX))
learning_rate_decay = np.array(0.3, dtype=theano.config.floatX)
updates = OrderedDict()

for name, layer in net.items():  
    layer_params = layer.get_params(trainable=True)
    if name in ['new_output', 'fc6', 'fc7']:
        layer_lr = learning_rate
    else:
        layer_lr = learning_rate / 10
    if name != 'fc8':
    	layer_updates = lasagne.updates.nesterov_momentum(loss, layer_params, learning_rate=layer_lr, momentum=0.9)
    	updates.update(layer_updates)

# Compile functions for training, validation and prediction
train_fn = theano.function([X_sym, y_sym], [loss, acc], updates=updates)
val_fn = theano.function([X_sym, y_sym], [loss, acc])
pred_fn = theano.function([X_sym], prediction)

X, y, ids = load_train_data()
X, y, ids = sklearn.utils.shuffle(X, y, ids, random_state=0)

# generator splitting an iterable into chunks of maximum length N
def batches(iterable, N):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == N:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# We need a fairly small batch size to fit a large network like this in GPU memory
BATCH_SIZE = 12

def train_batch(ix):
    return train_fn(X_tr[ix], y_tr[ix])

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])

kf = LabelKFold(ids, n_folds=8)

print(X.shape)

lasagne.layers.set_all_param_values(net['prob'], d['param values'])

for i, (tr_ix, val_ix) in enumerate(kf):
    print('CV Fold', i)
    X_tr = X[tr_ix]
    y_tr = y[tr_ix]
    X_val = X[val_ix]
    y_val = y[val_ix]

    net['new_output'] = DenseLayer(net['drop7'], num_units=8, nonlinearity=softmax, W=lasagne.init.Normal(0.01)) 
    lasagne.layers.set_all_param_values(net['prob'], d['param values'])
    learning_rate.set_value(0.0001)

    for epoch in range(1):

        kf2 = KFold(len(y_tr), n_folds=np.floor(len(y_tr)/BATCH_SIZE), shuffle=True, random_state=1)
        progbar = Progbar(np.floor(len(y_tr)/BATCH_SIZE))
        for j, (_, ix) in enumerate(kf2):
            loss, acc = train_batch(ix)
            progbar.add(1)

        learning_rate.set_value(learning_rate.get_value() * learning_rate_decay)

        v_ix = range(len(y_val))
        t_ix = range(len(y_tr))
        np.random.shuffle(v_ix)
        np.random.shuffle(t_ix)

        tr_loss_tot = 0.
        tr_acc_tot = 0
        val_loss_tot = 0.
        val_acc_tot = 0.

        for chunk in batches(t_ix[:2000], BATCH_SIZE):
            tr_loss, tr_acc = val_fn(X_tr[chunk], y_tr[chunk])
            tr_loss_tot += tr_loss * len(chunk)
            tr_acc_tot += tr_acc * len(chunk)

        for chunk in batches(v_ix, BATCH_SIZE):
            val_loss, val_acc = val_fn(X_val[chunk], y_val[chunk])
            val_loss_tot += val_loss * len(chunk)
            val_acc_tot += val_acc * len(chunk)

        tr_loss_tot /= len(t_ix[:2000])
        tr_acc_tot /= len(t_ix[:2000])

        val_loss_tot /= len(v_ix)
        val_acc_tot /= len(v_ix)

        res = [epoch, round(tr_loss_tot,3), round(tr_acc_tot*100,3), round(val_loss_tot,3), round(val_acc_tot * 100,3)]
        print('\t'.join(map(str,res)))
        #print(epoch, val_loss_tot, val_acc_tot * 100)

    # if loss is too high, convergence failure so don't bother to make predictions
    if val_loss_tot > 1.0:
        continue

    print("Predicting with test images...")
    imgs_per_batch = 99
    for j in range(0, 1001, imgs_per_batch):
        X_sub_part, sub_ids_part = load_test_data(j, imgs_per_batch)
        y_proba_part = pred_fn(X_sub_part)
        if j==0:
            y_proba = y_proba_part
            ids = sub_ids_part
        else:
            y_proba = np.append(y_proba, y_proba_part, axis=0)
            ids = np.append(ids, sub_ids_part, axis=0)
        if j%99==0:
            print(j)

    make_submission('../output/submission_vgg19_' + str(i) + '.csv', y_proba, ids)
