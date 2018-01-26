import pickle
import gc
import numpy as np
import lasagne, sklearn, theano
import theano.tensor as T
from collections import OrderedDict
from sklearn.cross_validation import train_test_split, LabelKFold, KFold
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import Pool2DLayer
from lasagne.layers import DenseLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers import ConcatLayer
from lasagne.layers.normalization import batch_norm
from lasagne.nonlinearities import softmax, linear
from lasagne.utils import floatX
from scipy.ndimage import rotate
from keras.utils.generic_utils import Progbar
from data_inception import load_train_data, load_test_data, make_submission

gc.collect()
np.random.seed(19)
IMG_SHAPE = 299, 299

def bn_conv(input_layer, **kwargs):
    l = Conv2DLayer(input_layer, **kwargs)
    l = batch_norm(l, epsilon=0.001)
    return l


def inceptionA(input_layer, nfilt):
    # Corresponds to a modified version of figure 5 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=5, pad=2)

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=3, pad=1)

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionB(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=3, pad=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionC(input_layer, nfilt):
    # Corresponds to figure 6 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 7), pad=(0, 3))
    l3 = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(7, 1), pad=(3, 0))
    l3 = bn_conv(l3, num_filters=nfilt[2][4], filter_size=(1, 7), pad=(0, 3))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode='average_exc_pad')
    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2, l3, l4])


def inceptionD(input_layer, nfilt):
    # Corresponds to a modified version of figure 10 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)
    l1 = bn_conv(l1, num_filters=nfilt[0][1], filter_size=3, stride=2)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2 = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 7), pad=(0, 3))
    l2 = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(7, 1), pad=(3, 0))
    l2 = bn_conv(l2, num_filters=nfilt[1][3], filter_size=3, stride=2)

    l3 = Pool2DLayer(input_layer, pool_size=3, stride=2)

    return ConcatLayer([l1, l2, l3])


def inceptionE(input_layer, nfilt, pool_mode):
    # Corresponds to figure 7 in the paper
    l1 = bn_conv(input_layer, num_filters=nfilt[0][0], filter_size=1)

    l2 = bn_conv(input_layer, num_filters=nfilt[1][0], filter_size=1)
    l2a = bn_conv(l2, num_filters=nfilt[1][1], filter_size=(1, 3), pad=(0, 1))
    l2b = bn_conv(l2, num_filters=nfilt[1][2], filter_size=(3, 1), pad=(1, 0))

    l3 = bn_conv(input_layer, num_filters=nfilt[2][0], filter_size=1)
    l3 = bn_conv(l3, num_filters=nfilt[2][1], filter_size=3, pad=1)
    l3a = bn_conv(l3, num_filters=nfilt[2][2], filter_size=(1, 3), pad=(0, 1))
    l3b = bn_conv(l3, num_filters=nfilt[2][3], filter_size=(3, 1), pad=(1, 0))

    l4 = Pool2DLayer(
        input_layer, pool_size=3, stride=1, pad=1, mode=pool_mode)

    l4 = bn_conv(l4, num_filters=nfilt[3][0], filter_size=1)

    return ConcatLayer([l1, l2a, l2b, l3a, l3b, l4])


def build_network():
    net = {}

    net['input'] = InputLayer((None, 3, 299, 299))
    net['conv'] = bn_conv(net['input'],
                          num_filters=32, filter_size=3, stride=2)
    net['conv_1'] = bn_conv(net['conv'], num_filters=32, filter_size=3)
    net['conv_2'] = bn_conv(net['conv_1'],
                            num_filters=64, filter_size=3, pad=1)
    net['pool'] = Pool2DLayer(net['conv_2'], pool_size=3, stride=2, mode='max')

    net['conv_3'] = bn_conv(net['pool'], num_filters=80, filter_size=1)

    net['conv_4'] = bn_conv(net['conv_3'], num_filters=192, filter_size=3)

    net['pool_1'] = Pool2DLayer(net['conv_4'],
                                pool_size=3, stride=2, mode='max')
    net['mixed/join'] = inceptionA(
        net['pool_1'], nfilt=((64,), (48, 64), (64, 96, 96), (32,)))
    net['mixed_1/join'] = inceptionA(
        net['mixed/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_2/join'] = inceptionA(
        net['mixed_1/join'], nfilt=((64,), (48, 64), (64, 96, 96), (64,)))

    net['mixed_3/join'] = inceptionB(
        net['mixed_2/join'], nfilt=((384,), (64, 96, 96)))

    net['mixed_4/join'] = inceptionC(
        net['mixed_3/join'],
        nfilt=((192,), (128, 128, 192), (128, 128, 128, 128, 192), (192,)))

    net['mixed_5/join'] = inceptionC(
        net['mixed_4/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_6/join'] = inceptionC(
        net['mixed_5/join'],
        nfilt=((192,), (160, 160, 192), (160, 160, 160, 160, 192), (192,)))

    net['mixed_7/join'] = inceptionC(
        net['mixed_6/join'],
        nfilt=((192,), (192, 192, 192), (192, 192, 192, 192, 192), (192,)))

    net['mixed_8/join'] = inceptionD(
        net['mixed_7/join'],
        nfilt=((192, 320), (192, 192, 192, 192)))

    net['mixed_9/join'] = inceptionE(
        net['mixed_8/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='average_exc_pad')

    net['mixed_10/join'] = inceptionE(
        net['mixed_9/join'],
        nfilt=((320,), (384, 384, 384), (448, 384, 384, 384), (192,)),
        pool_mode='max')

    net['pool3'] = GlobalPoolLayer(net['mixed_10/join'])

    net['softmax'] = DenseLayer(
        net['pool3'], num_units=1008, nonlinearity=softmax)

    return net

# Load model weights and metadata
d = pickle.load(open('../input/pretrained/inception_v3.pkl'))

# Build the network and fill with pretrained weights
net = build_network()

# Define loss function and metrics, and get an updates dictionary
X_sym = T.tensor4()
y_sym = T.ivector()

# We'll connect our output classifier to the last fully connected layer of the network
net['new_output'] = DenseLayer(net['pool3'], num_units=8, nonlinearity=softmax, W=lasagne.init.Normal(0.01)) 

prediction = lasagne.layers.get_output(net['new_output'], X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym), dtype=theano.config.floatX)

learning_rate = theano.shared(np.array(0.005, dtype=theano.config.floatX))
learning_rate_decay = np.array(0.5, dtype=theano.config.floatX)
updates = OrderedDict()

for name, layer in net.items(): 
    print(name) 
    layer_params = layer.get_params(trainable=True)
    if name in ['new_output']:
        layer_lr = learning_rate
    else:
        layer_lr = learning_rate / 10
    if name != 'softmax':
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

lasagne.layers.set_all_param_values(net['softmax'], d['param values'])

for i, (tr_ix, val_ix) in enumerate(kf):
    print('CV Fold', i)
    X_tr = X[tr_ix]
    y_tr = y[tr_ix]
    X_val = X[val_ix]
    y_val = y[val_ix]

    net['new_output'] = DenseLayer(net['pool3'], num_units=8, nonlinearity=softmax, W=lasagne.init.Normal(0.01)) 
    lasagne.layers.set_all_param_values(net['softmax'], d['param values'])
    learning_rate.set_value(0.005)

    for epoch in range(3):

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

    make_submission('../output/submission_ggnet_' + str(i) + '.csv', y_proba, ids)
