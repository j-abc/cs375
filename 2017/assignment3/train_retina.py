from __future__ import division, print_function, absolute_import
import os, sys
from collections import OrderedDict
import numpy as np

import tensorflow as tf

from tfutils import base, data, model, optimizer, utils
from deepretina.metrics import cc
import copy
from layers import conv, fc, gaussian_noise_layer
# group 6
seed = 6

# toggle this to train or to validate at the end
train_net = True
# toggle this to train on whitenoise or naturalscene data
stim_type = 'whitenoise'
#stim_type = 'naturalscene'
# Figure out the hostname
host = os.uname()[1]
#if 'neuroaicluster' in host:
if True: # the line above seemed useless
    if train_net:
        print('In train mode...')
        TOTAL_BATCH_SIZE = 5000
        MB_SIZE = 5000
        NUM_GPUS = 2
    else:
        print('In val mode...')
        if stim_type == 'whitenoise':
            TOTAL_BATCH_SIZE = 5957
            MB_SIZE = 5957
            NUM_GPUS = 2
        else:
            TOTAL_BATCH_SIZE = 5956
            MB_SIZE = 5956
            NUM_GPUS = 2

else:
    print("Data path not found!!")
    exit()

if not isinstance(NUM_GPUS, list):
    DEVICES = ['/gpu:' + str(i) for i in range(NUM_GPUS)]
else:
    DEVICES = ['/gpu:' + str(i) for i in range(len(NUM_GPUS))]

MODEL_PREFIX = 'model_0'

# Data parameters
if stim_type == 'whitenoise':
    N_TRAIN = 323762
    N_TEST = 5957
else:
    N_TRAIN = 323756
    N_TEST = 5956

INPUT_BATCH_SIZE = int(1024 / 2) # queue size
OUTPUT_BATCH_SIZE = TOTAL_BATCH_SIZE
print('TOTAL BATCH SIZE:', OUTPUT_BATCH_SIZE)
NUM_BATCHES_PER_EPOCH = N_TRAIN // OUTPUT_BATCH_SIZE
IMAGE_SIZE_RESIZE = 50

NCELLS = 5

DATA_PATH = '/datasets/deepretina_data/tf_records/' + stim_type
WHITE_DATA_PATH = '/datasets/deepretina_data/tf_records/whitenoise'
NATURAL_DATA_PATH = '/datasets/deepretina_data/tf_records/naturalscene'
print('Data path: ', DATA_PATH)

# data provider
class retinaTF(data.TFRecordsParallelByFileProvider):

  def __init__(self,
               source_dirs,
               resize=IMAGE_SIZE_RESIZE,
               **kwargs
               ):

    if resize is None:
      self.resize = 50
    else:
      self.resize = resize

    postprocess = {'images': [], 'labels': []}
    postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
    postprocess['images'].insert(1, (tf.reshape, ([-1] + [50, 50, 40], ), {}))
    postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 

    postprocess['labels'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
    postprocess['labels'].insert(1, (tf.reshape, ([-1] + [5], ), {}))

    super(retinaTF, self).__init__(
      source_dirs,
      postprocess=postprocess,
      **kwargs
    )


  def postproc_imgs(self, ims):
    def _postprocess_images(im):
        im = tf.image.resize_images(im, [self.resize, self.resize])
        return im
    return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

def ln(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, seed=0, cfg_final=None):
    params = OrderedDict()
    input_shape = inputs['images'].get_shape().as_list()
    batch_size = input_shape[0]
    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size

    # implement your LN model here
    # Use SGD
    # Optimizer = Adam, learning rate of 1e-3
    # L2 regularization = 1e-3 on the fully connected layer weights
    #  w = [5,100000]
    # b = [5]
    
    xf = tf.reshape(inputs['images'], [-1,100000])
    b = tf.get_variable(shape=[5], dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='b') 
    # Should shape be transposed like this?
    w = tf.get_variable(shape=[100000,5], dtype=tf.float32, regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='w')
    out = tf.matmul(xf, w) + b      
    
    # Use softplus linearity to ensure nonnegative firing rates
    out = tf.nn.softplus(out)
    outputs = {}
    outputs['pred'] = out
    return outputs, params

def cnn(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, seed=0, cfg_final=None):
    params = OrderedDict()
    input_shape = inputs['images'].get_shape().as_list()
    batch_size = input_shape[0]
    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size

    #start
    outputs = inputs
    # first conv layer
    outputs['conv1'] = conv(outputs['images'], 16, 15, name = 'conv1',
        padding = 'VALID', batch_norm = False, weight_decay = 1e-3)
    # gaussian noise
    if train:
        outputs['conv1'] = gaussian_noise_layer(outputs['conv1'], 0.1)
    # second layer
    outputs['conv2'] = conv(outputs['conv1'], 8, 9, name = 'conv2',
        padding = 'VALID', batch_norm = False, weight_decay = 1e-3)
    # gaussian noise
    if train:
        outputs['conv2'] = gaussian_noise_layer(outputs['conv2'], 0.1)
    # final fc layer
    outputs['pred'] = fc(outputs['conv2'], 5, name = 'fc1',
        weight_decay = 1e-3, activation = 'softplus')
    # just add regularization
    regularize_activity = tf.contrib.layers.l1_regularizer (1e-3)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, regularize_activity(outputs['pred']))
    # end
    return outputs, params

def poisson_loss(outputs, inputs):
    # epsilon
    epsilon = tf.constant(1e-8)
    # implement the poisson loss here
    # K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)
    loss = tf.reduce_mean(outputs['pred'] - inputs * tf.log(outputs['pred'] + epsilon), axis=-1)
    #loss = tf.py_func(cc, [inputs['labels'], outputs['pred']], tf.float32)
    return loss

def mean_loss_with_reg(loss):
    return tf.reduce_mean(loss) + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

def online_agg(agg_res, res, step):
    special_k = ['pred', 'labels']
    if agg_res is None:
        agg_res = {k: [] if k not in special_k else None for k in res}
    for k, v in res.items():
        if k not in special_k:
            agg_res[k].append(np.mean(v))
        else:
            if agg_res[k] is None:
                agg_res[k] = v
            else:
                agg_res[k] = np.concatenate((agg_res[k], v), axis=0)
    return agg_res

def my_online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(v)
    return agg_res

def get_pearson(pred, truth):
    y_hat_mu = pred - np.mean(pred, axis=0, keepdims=True)
    y_mu = truth - np.mean(truth, axis=0, keepdims=True)    
    y_hat_std = np.std(pred, axis=0, keepdims=True)
    y_std = np.std(truth, axis=0, keepdims=True)
    corr = np.mean(y_mu * y_hat_mu, axis=0, keepdims=True)/(y_std * y_hat_std)
    return corr

def pearson_agg(results):
    for k,v in results.iteritems():
        results[k] = np.concatenate(v, axis=0)
        
    testcorrs = {}
    testcorrs['corr'] = get_pearson(results['pred'], results['label'])
       
    return testcorrs

def loss_metric(inputs, outputs, target, **kwargs):
    metrics_dict = {}
    metrics_dict['poisson_loss'] = mean_loss_with_reg(poisson_loss(outputs=outputs, inputs=inputs[target]), **kwargs)
    return metrics_dict

def get_targets(inputs, outputs, target, **kwargs):
    targets_dict = {}
    targets_dict['pred'] = outputs['pred']
    targets_dict['label'] = inputs[target]
    
    return targets_dict

def mean_losses_keep_rest(step_results):
    retval = {}
    keys = step_results[0].keys()
    print('KEYS: ', keys)
    for k in keys:
        plucked = [d[k] for d in step_results]
        if isinstance(k, str) and 'loss' in k:
            retval[k] = np.mean(plucked)
        else:
            retval[k] = plucked
    return retval

# model parameters

default_params = {
    'save_params': {
        'host': '10.138.0.3',
        'port': 24444,
        'dbname': 'deepretina',
        'collname': stim_type,
        'exp_id': 'trainval0',

        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 50,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 50,
        'save_filters_freq': 50,
        'cache_filters_freq': 50,
        # 'cache_dir': None,  # defaults to '~/.tfutils'
    },

    'load_params': {
        'do_restore': True,
        'query': None
    },

    'model_params': {
        'func': ln,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    },

    'train_params': {
        'minibatch_size': MB_SIZE,
        'data_params': {
            'func': retinaTF,
            'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
            'resize': IMAGE_SIZE_RESIZE,
            'batch_size': INPUT_BATCH_SIZE,
            'file_pattern': 'train*.tfrecords',
            'n_threads': 4
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': OUTPUT_BATCH_SIZE,
            'capacity': 11*INPUT_BATCH_SIZE,
            'min_after_dequeue': 10*INPUT_BATCH_SIZE,
            'seed': seed,
        },
        'thres_loss': float('inf'),
        'num_steps': 50 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'validate_first': True,
    },

    'loss_params': {
        'targets': ['labels'],
        'agg_func': mean_loss_with_reg,
        'loss_per_case_func': poisson_loss,
        'loss_per_case_func_params' : {
                '_outputs': 'outputs', 
                '_targets_$all': 'inputs',
        },
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 1e-3,
        'decay_rate': 1.0, # constant learning rate
        'decay_steps': NUM_BATCHES_PER_EPOCH,
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.AdamOptimizer,
        'clip': True,
        'trainable_names': None
    },

    'validation_params': {
        'white_noise_testcorr': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join('/datasets/deepretina_data/tf_records/whitenoise', 'images'), os.path.join('/datasets/deepretina_data/tf_records/whitenoise', 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 1
            },
            'targets': {
                'func': get_targets,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': 5957 // MB_SIZE + 1,
            'agg_func': pearson_agg,   # lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': my_online_agg
        },
        'natural_scenes_testcorr': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join('/datasets/deepretina_data/tf_records/naturalscene', 'images'), os.path.join('/datasets/deepretina_data/tf_records/naturalscene', 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 1
            },
            'targets': {
                'func': get_targets,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': 5956 // MB_SIZE + 1,
            'agg_func': pearson_agg,   # lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': my_online_agg
        },
        #'train_loss': {
        #    'data_params': {
        #        'func': retinaTF,
        #        'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
        #        'resize': IMAGE_SIZE_RESIZE,
        #        'batch_size': INPUT_BATCH_SIZE,
        #        'file_pattern': 'train*.tfrecords',
        #        'n_threads': 4
        #    },
        #    'targets': {
        #        'func': loss_metric,
        #        'target': 'labels',
        #    },
        #    'queue_params': {
        #        'queue_type': 'fifo',
        #        'batch_size': MB_SIZE,
        #        'capacity': 11*INPUT_BATCH_SIZE,
        #        'min_after_dequeue': 10*INPUT_BATCH_SIZE,
        #        'seed': seed,
        #    },
        #    'num_steps': N_TRAIN // OUTPUT_BATCH_SIZE + 1,
        #    'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
        #    'online_agg_func': online_agg
        #}
    },
    'log_device_placement': False,  # if variable placement has to be logged
}


def get_stim_params(stim_type):
    if stim_type == 'whitenoise':
        N_TRAIN = 323762
        N_TEST = 5957
    else:
        N_TRAIN = 323756
        N_TEST = 5956
    return {
        'N_TRAIN': N_TRAIN,
        'DATA_PATH': '/datasets/deepretina_data/tf_records/' + stim_type
    }



def train_ln(stim_type = 'whitenoise'):
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'ln_model'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    params['model_params']['func'] = ln
    params['learning_rate_params']['learning_rate'] = 1e-3

    # custom crap for the train stim type
    stim_params = get_stim_params(stim_type)
    params['train_params']['data_params']['source_dirs'] = [os.path.join(stim_params['DATA_PATH'], 'images'), os.path.join(stim_params['DATA_PATH'], 'labels')]
    #params['validation_params']['train_loss']['data_params']['source_dirs'] = [os.path.join(stim_params['DATA_PATH'], 'images'), os.path.join(stim_params['DATA_PATH'], 'labels')]
    NUM_BATCHES_PER_EPOCH = stim_params['N_TRAIN'] // OUTPUT_BATCH_SIZE
    params['train_params']['num_steps'] = 50 * NUM_BATCHES_PER_EPOCH
    params['learning_rate_params']['decay_steps'] = NUM_BATCHES_PER_EPOCH
    base.train_from_params(**params)

def train_cnn(stim_type = 'whitenoise'):
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'cnn'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval1'

    params['model_params']['func'] = cnn
    params['learning_rate_params']['learning_rate'] = 1e-3

    # custom crap for the train stim type
    stim_params = get_stim_params(stim_type)
    params['train_params']['data_params']['source_dirs'] = [os.path.join(stim_params['DATA_PATH'], 'images'), os.path.join(stim_params['DATA_PATH'], 'labels')] 
    #params['validation_params']['train_loss']['data_params']['source_dirs'] = [os.path.join(stim_params['DATA_PATH'], 'images'), os.path.join(stim_params['DATA_PATH'], 'labels')] 
    NUM_BATCHES_PER_EPOCH = stim_params['N_TRAIN'] // OUTPUT_BATCH_SIZE
    params['train_params']['num_steps'] = 50 * NUM_BATCHES_PER_EPOCH
    params['learning_rate_params']['decay_steps'] = NUM_BATCHES_PER_EPOCH
    base.train_from_params(**params)
 
if __name__ == '__main__':
    #train_cnn()
    train_ln()

