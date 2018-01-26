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
from sklearn.cross_validation import train_test_split, LabelKFold, KFold
from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
from lasagne.utils import floatX
from scipy.ndimage import rotate
from keras.utils.generic_utils import Progbar
from data_pretrained import load_train_data, load_test_data, make_submission

gc.collect()
np.random.seed(19)
IMG_SHAPE = 224, 224

def build_inception_module(name, input_layer, nfilters):
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(net['pool'], nfilters[0], 1, flip_filters=False)
    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)
    net['3x3_reduce'] = ConvLayer(input_layer, nfilters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)
    net['5x5_reduce'] = ConvLayer(input_layer, nfilters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)
    net['output'] = ConcatLayer([net['1x1'],net['3x3'],net['5x5'],net['pool_proj'],])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}


def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, None, None))
    net['conv1/7x7_s2'] = ConvLayer(net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = PoolLayer(net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_3a',net['pool2/3x3_s2'],[32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',net['inception_3a/output'],[64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_4a',net['pool3/3x3_s2'],[64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',net['inception_4a/output'],[64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',net['inception_4b/output'],[64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',net['inception_4c/output'],[64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',net['inception_4d/output'],[128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_5a',net['pool4/3x3_s2'],[128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',net['inception_5a/output'],[128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],num_units=1000,nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'],nonlinearity=softmax)
    return net

# Load model weights and metadata
d = pickle.load(open('../input/pretrained/blvc_googlenet.pkl'))

# Build the network and fill with pretrained weights
net = build_model()

# Define loss function and metrics, and get an updates dictionary
X_sym = T.tensor4()
y_sym = T.ivector()

# We'll connect our output classifier to the last fully connected layer of the network
net['new_output'] = DenseLayer(net['pool5/7x7_s1'], num_units=10, nonlinearity=softmax, W=lasagne.init.HeNormal(0.01)) 

prediction = lasagne.layers.get_output(net['new_output'], X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym), dtype=theano.config.floatX)

learning_rate = theano.shared(np.array(0.0002, dtype=theano.config.floatX))
learning_rate_decay = np.array(0.1, dtype=theano.config.floatX)
updates = OrderedDict()

for name, layer in net.items(): 
    print(name) 
    layer_params = layer.get_params(trainable=True)
    if name in ['new_output', 'fc6', 'fc7']:
        layer_lr = learning_rate
    else:
        layer_lr = learning_rate #/ 10
    if name != 'loss3/classifier':
    	layer_updates = lasagne.updates.nesterov_momentum(loss, layer_params, learning_rate=layer_lr, momentum=0.9)
    	updates.update(layer_updates)

# Compile functions for training, validation and prediction
train_fn = theano.function([X_sym, y_sym], [loss, acc], updates=updates)
val_fn = theano.function([X_sym, y_sym], [loss, acc])
pred_fn = theano.function([X_sym], prediction)

X, y, drivers = load_train_data('../input', grayscale=False, img_shape=IMG_SHAPE, usecache=True)
X, y, drivers = sklearn.utils.shuffle(X, y, drivers, random_state=0)

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

kf = LabelKFold(drivers, n_folds=8)
lasagne.layers.set_all_param_values(net['prob'], d['param values'])
for i, (tr_ix, val_ix) in enumerate(kf):
    print('CV Fold', i)
    X_tr = X[tr_ix]
    y_tr = y[tr_ix]
    X_val = X[val_ix]
    y_val = y[val_ix]

    #net['new_output'] = DenseLayer(net['pool5/7x7_s1'], num_units=10, nonlinearity=softmax, W=lasagne.init.Normal(0.01)) 
    lasagne.layers.set_all_param_values(net['prob'], d['param values'])
    learning_rate.set_value(0.0002)

    for epoch in range(2):

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
    imgs_per_batch = 100
    for j in range(0, 79726, imgs_per_batch):
        X_sub_part, sub_ids_part = load_test_data('../input', img_shape=IMG_SHAPE, img_start_ix=j, max_img=imgs_per_batch)
        y_proba_part = pred_fn(X_sub_part)
        if j==0:
            y_proba = y_proba_part
            ids = sub_ids_part
        else:
            y_proba = np.append(y_proba, y_proba_part, axis=0)
            ids = np.append(ids, sub_ids_part, axis=0)
        if j%1000==0:
            print(j)

    make_submission('../output/submission_ggnet_' + str(i) + '.csv', y_proba, ids)
