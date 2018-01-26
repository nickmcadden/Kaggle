import pickle
import gc
import numpy as np
import lasagne, sklearn, theano
import theano.tensor as T
from collections import OrderedDict
from sklearn.cross_validation import train_test_split, KFold, StratifiedKFold, LabelKFold
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer
from lasagne.layers import BatchNormLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import ElemwiseSumLayer
from lasagne.layers import DenseLayer
from lasagne.nonlinearities import rectify, softmax
from lasagne.utils import floatX
from scipy.ndimage import rotate
from keras.utils.generic_utils import Progbar
from data_jitter import load_train_data, load_test_data, make_submission, rotatebatch

gc.collect()
np.random.seed(1)
IMG_SHAPE = 224, 224

def build_simple_block(incoming_layer, names,
                       num_filters, filter_size, stride, pad,
                       use_bias=False, nonlin=rectify):
    """Creates stacked Lasagne layers ConvLayer -> BN -> (ReLu)
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    names : list of string
        Names of the layers in block
    num_filters : int
        Number of filters in convolution layer
    filter_size : int
        Size of filters in convolution layer
    stride : int
        Stride of convolution layer
    pad : int
        Padding of convolution layer
    use_bias : bool
        Whether to use bias in conlovution layer
    nonlin : function
        Nonlinearity type of Nonlinearity layer
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    net = []
    net.append((
            names[0],
            ConvLayer(incoming_layer, num_filters, filter_size, pad, stride,
                      flip_filters=False, nonlinearity=None) if use_bias
            else ConvLayer(incoming_layer, num_filters, filter_size, stride, pad, b=None,
                           flip_filters=False, nonlinearity=None)
        ))

    net.append((
            names[1],
            BatchNormLayer(net[-1][1])
        ))
    if nonlin is not None:
        net.append((
            names[2],
            NonlinearityLayer(net[-1][1], nonlinearity=nonlin)
        ))

    return dict(net), net[-1][0]


def build_residual_block(incoming_layer, ratio_n_filter=1.0, ratio_size=1.0, has_left_branch=False,
                         upscale_factor=4, ix=''):
    """Creates two-branch residual block
    Parameters:
    ----------
    incoming_layer : instance of Lasagne layer
        Parent layer
    ratio_n_filter : float
        Scale factor of filter bank at the input of residual block
    ratio_size : float
        Scale factor of filter size
    has_left_branch : bool
        if True, then left branch contains simple block
    upscale_factor : float
        Scale factor of filter bank at the output of residual block
    ix : int
        Id of residual block
    Returns
    -------
    tuple: (net, last_layer_name)
        net : dict
            Dictionary with stacked layers
        last_layer_name : string
            Last layer name
    """
    simple_block_name_pattern = ['res%s_branch%i%s', 'bn%s_branch%i%s', 'res%s_branch%i%s_relu']

    net = {}

    # right branch
    net_tmp, last_layer_name = build_simple_block(
        incoming_layer, map(lambda s: s % (ix, 2, 'a'), simple_block_name_pattern),
        int(lasagne.layers.get_output_shape(incoming_layer)[1]*ratio_n_filter), 1, int(1.0/ratio_size), 0)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'b'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1], 3, 1, 1)
    net.update(net_tmp)

    net_tmp, last_layer_name = build_simple_block(
        net[last_layer_name], map(lambda s: s % (ix, 2, 'c'), simple_block_name_pattern),
        lasagne.layers.get_output_shape(net[last_layer_name])[1]*upscale_factor, 1, 1, 0,
        nonlin=None)
    net.update(net_tmp)

    right_tail = net[last_layer_name]
    left_tail = incoming_layer

    # left branch
    if has_left_branch:
        net_tmp, last_layer_name = build_simple_block(
            incoming_layer, map(lambda s: s % (ix, 1, ''), simple_block_name_pattern),
            int(lasagne.layers.get_output_shape(incoming_layer)[1]*4*ratio_n_filter), 1, int(1.0/ratio_size), 0,
            nonlin=None)
        net.update(net_tmp)
        left_tail = net[last_layer_name]

    net['res%s' % ix] = ElemwiseSumLayer([left_tail, right_tail], coeffs=1)
    net['res%s_relu' % ix] = NonlinearityLayer(net['res%s' % ix], nonlinearity=rectify)

    return net, 'res%s_relu' % ix


def build_model():
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    sub_net, parent_layer_name = build_simple_block(net['input'], ['conv1', 'bn_conv1', 'conv1_relu'],64, 7, 3, 2, use_bias=True)
    net.update(sub_net)
    net['pool1'] = PoolLayer(net[parent_layer_name], pool_size=3, stride=2, pad=0, mode='max', ignore_border=False)
    block_size = list('abc')
    parent_layer_name = 'pool1'
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1, 1, True, 4, ix='2%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='2%s' % c)
        net.update(sub_net)

    block_size = list('abcd')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='3%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='3%s' % c)
        net.update(sub_net)

    block_size = list('abcdef')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='4%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='4%s' % c)
        net.update(sub_net)

    block_size = list('abc')
    for c in block_size:
        if c == 'a':
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/2, 1.0/2, True, 4, ix='5%s' % c)
        else:
            sub_net, parent_layer_name = build_residual_block(net[parent_layer_name], 1.0/4, 1, False, 4, ix='5%s' % c)
        net.update(sub_net)
    net['pool5'] = PoolLayer(net[parent_layer_name], pool_size=7, stride=1, pad=0, mode='average_exc_pad', ignore_border=False)
    net['fc1000'] = DenseLayer(net['pool5'], num_units=1000, nonlinearity=None)
    net['prob'] = NonlinearityLayer(net['fc1000'], nonlinearity=softmax)

    return net

# Load model weights and metadata
d = pickle.load(open('../input/pretrained/resnet50.pkl'))

# Build the network and fill with pretrained weights
net = build_model()

# Define loss function and metrics, and get an updates dictionary
X_sym = T.tensor4()
y_sym = T.ivector()

# We'll connect our output classifier to the last fully connected layer of the network
net['new_output'] = DenseLayer(net['pool5'], num_units=8, nonlinearity=softmax, W=lasagne.init.Normal(0.01)) 

prediction = lasagne.layers.get_output(net['new_output'], X_sym)
loss = lasagne.objectives.categorical_crossentropy(prediction, y_sym)
loss = loss.mean()

acc = T.mean(T.eq(T.argmax(prediction, axis=1), y_sym), dtype=theano.config.floatX)

learning_rate = theano.shared(np.array(0.001, dtype=theano.config.floatX))
learning_rate_decay = np.array(0.3, dtype=theano.config.floatX)
updates = OrderedDict()

print("Setting learning rates...")
for name, layer in net.items():  
    print(name)
    layer_params = layer.get_params(trainable=True)
    if name in ['new_output', 'fc1000']:
        layer_lr = learning_rate
    else:
        layer_lr = learning_rate / 10
    if name != 'fc1000':
    	layer_updates = lasagne.updates.nesterov_momentum(loss, layer_params, learning_rate=layer_lr, momentum=0.9)
    	updates.update(layer_updates)

# Compile functions for training, validation and prediction
print("Compiling theano functions...")
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
    X_temp = rotatebatch(X_tr[ix])
    return train_fn(X_temp, y_tr[ix])

def val_batch():
    ix = range(len(y_val))
    np.random.shuffle(ix)
    ix = ix[:BATCH_SIZE]
    return val_fn(X_val[ix], y_val[ix])

kf = LabelKFold(ids, n_folds=8)

print(X.shape)

lasagne.layers.set_all_param_values(net['prob'], d['values'])

for i, (tr_ix, val_ix) in enumerate(kf):
    print('CV Fold', i)
    X_tr = X[tr_ix]
    y_tr = y[tr_ix]
    X_val = X[val_ix]
    y_val = y[val_ix]

    #net['new_output'] = DenseLayer(net['pool5'], num_units=8, nonlinearity=softmax, W=lasagne.init.Normal(0.01)) 
    lasagne.layers.set_all_param_values(net['prob'], d['values'])
    learning_rate.set_value(0.001)

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

    make_submission('../output/submission_resnet50_' + str(i) + '.csv', y_proba, ids)
