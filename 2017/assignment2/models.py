import os
import numpy as np
import tensorflow as tf
from color_utils import preprocess, decode
from losses import *

'''
TODO: 
- change the loss to be _out_ of the model, the model should just return 
the values at each layer. Specify in a different function the loss functions
- merge autoencoder and var-autoencoder: just return a parameter if you want to make
variational
- add the colorization model
'''

# variational autoencoder
def vae_model(inputs, train=True, norm=True, **kwargs):
    # create outputs
    # get input shape
    in_shp = inputs['images'].get_shape().as_list()
    outputs = inputs
    encoding_layernames = ['enc' + str(i) for i in range(1,4)]
    encoding_ksizes = [3, 3, 3]
    encoding_strides = [2, 2, 2]
    encoding_channels = [32, 64, 128]
    encoding_bns = [False, True, False]

    decoding_layernames = ['dec' + str(i) for i in range(1,4)]
    decoding_ksizes = [3, 3, 3]
    decoding_strides = [2, 2, 2]
    decoding_channels = [64, 32, in_shp[-1]]
    decoding_bns = [True, False, True]

    zdim = 30
    weight_decay = 1e-3
    dropout = .5 if train else None
    # encoding layers
    current_layer = outputs['images']
    #print current_layer.get_shape().as_list()
    for i,layer_name in enumerate(encoding_layernames):
        outputs[layer_name], outputs[layer_name + '_kernel'] = conv(
            current_layer,
            encoding_channels[i],
            ksize=encoding_ksizes[i],
            strides=encoding_strides[i],
            padding='SAME',
            weight_decay=weight_decay,
            name=layer_name,
            layer = layer_name,
            batch_norm = encoding_bns[i],
            )
        current_layer = outputs[layer_name]
        #print current_layer.get_shape().as_list()
    # z, and sigma_z
    outputs['z_mean'] = fc(current_layer,
        zdim,
        bias=1,
        weight_decay=weight_decay,
        activation=None,
        batch_norm=False,
        name='z_mean',
        layer='z_mean',
        dropout = dropout,
    )
    outputs['z_logstd'] = fc(current_layer,
        zdim,
        weight_decay=weight_decay,
        activation=None,
        batch_norm=False,
        name='z_logstd',
        layer='z_logstd',
        dropout = dropout,
    )
    # reparameterization trick
    noise = tf.random_normal([1, zdim])
    outputs['latent_encoding'] = tf.add(outputs['z_mean'], tf.multiply(noise, tf.exp(.5*outputs['z_logstd'])), name='latent_encoding')
    current_layer = outputs['latent_encoding']
    #print current_layer.get_shape().as_list()
    # decoding layers
    ## start by the last encoding layer outshape
    out_shp = outputs[encoding_layernames[-1]].get_shape().as_list()
    dec0 = fc(current_layer, out_shp[1]*out_shp[2]*out_shp[3],
        weight_decay=weight_decay,
        activation=None,
        batch_norm=False,
        name='dec0',
        layer='dec0',
    )
    outputs['dec0'] = tf.reshape(dec0, out_shp)
    current_layer = outputs['dec0']
    for i,layer_name in enumerate(decoding_layernames):
        outputs[layer_name], outputs[layer_name + '_kernel'] = conv(
            current_layer,
            decoding_channels[i],
            ksize=decoding_ksizes[i],
            strides=decoding_strides[i],
            padding='SAME',
            weight_decay=weight_decay,
            name=layer_name,
            layer = layer_name,
            batch_norm = decoding_bns[i],
            deconv = True,
            )
        current_layer = outputs[layer_name]
        #print current_layer.get_shape().as_list()
    # y
    outputs['pred'] = tf.nn.sigmoid(current_layer)
    #print outputs['pred'].get_shape().as_list()
    return outputs, {}

def colorful_model(inputs, train=True, norm=True, **kwargs):
    """
    colorful model from Zhang et al. 2016
    """
    # preprocess images
    data_l, gt_ab_313, prior_boost_nongray = tf.py_func(preprocess, [inputs['images']], [tf.float32,tf.float32,tf.float32])
    shp = inputs['images'].get_shape().as_list()
    print shp,(shp[0],shp[1]/4,shp[2]/4,313)
    data_l.set_shape((shp[0],shp[1],shp[2],1))
    gt_ab_313.set_shape((shp[0],shp[1]/4,shp[2]/4,313))
    # propagate input targets
    outputs = inputs
    outputs['data_l'] = data_l
    outputs['gt_ab_313'] = gt_ab_313
    dropout = .5 if train else None
    input_to_network = data_l
    weight_decay = 1e-3
    # setup
    layer_names = ['conv1_1','conv1_2','conv2_1',
    'conv2_2','conv3_1','conv3_2','conv3_3','conv4_1',
    'conv4_2','conv4_3','conv5_1','conv5_2','conv5_3',
    'conv6_1','conv6_2','conv6_3','conv7_1','conv7_2',
    'conv7_3','conv8_1','conv8_2','conv8_3']

    channels = [64,64,128,128,256,256,256,512,512,512,
    512,512,512,512,512,512,256,256,256,128,128,128,]

    strides = [1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, .5, 1, 1,]

    batch_norms = [False,True,False,True,False,
    False,True,False,False,True,False,False,True,False,
    False,True,False,False,True,False,False,False,]

    dilations = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,1,1,
    1,1,1,1,]

    current_layer = input_to_network

    for i in range(len(layer_names)):
        if strides[i] < 1:
            deconv = True
            ksize = 4
            strides[i] = int(round(1 / strides[i]))
            print strides[i]
        else:
            deconv = False
            ksize = 3
        outputs[layer_names[i]], outputs[layer_names[i] + '_kernel'] = conv(
            current_layer,
            channels[i],
            ksize=ksize,
            strides=strides[i],
            padding='SAME',
            weight_decay=weight_decay,
            name=layer_names[i],
            layer = layer_names[i],
            batch_norm = batch_norms[i],
            dilation = dilations[i],
            deconv = deconv,
            )
        current_layer = outputs[layer_names[i]]
        print current_layer.get_shape().as_list(), strides

    outputs['conv8_313'], outputs['conv8_313' + '_kernel'] = conv(
            current_layer,
            313,
            ksize=1,
            strides=1,
            padding='SAME',
            weight_decay=weight_decay,
            name='conv8_313',
            layer = 'conv8_313',
            deconv = False,
            activation = None,
            )

    outputs['pred'] = tf.py_func(decode, [data_l, outputs['conv8_313']], [tf.float32])

    return outputs, {}


def max_pool(x, ksize, strides,  name='pool', padding='SAME', layer = None):
    with tf.variable_scope(layer):
        if isinstance(ksize, int):
            ksize = [ksize, ksize]
        if isinstance(strides, int):
            strides = [1, strides, strides, 1]
    return tf.nn.max_pool(x, ksize= [1, ksize[0], ksize[1],1],
                        strides = strides,
                        padding = padding, name = name)

def fc(inp,
       out_depth,
       kernel_init='xavier',
       kernel_init_kwargs=None,
       bias=1,
       weight_decay=None,
       activation='relu',
       batch_norm=True,
       dropout=None,
       dropout_seed=None,
       name='fc',
       layer='blah'):
    with tf.variable_scope(layer):
        if weight_decay is None:
            weight_decay = 0.
        # assert out_shape is not None
        if kernel_init_kwargs is None:
            kernel_init_kwargs = {}
        resh = tf.reshape(inp, [inp.get_shape().as_list()[0], -1], name='reshape')
        in_depth = resh.get_shape().as_list()[-1]

        # weights
        init = initializer(kernel_init, **kernel_init_kwargs)
        kernel = tf.get_variable(initializer=init,
                                shape=[in_depth, out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='weights')
        init = initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')

        # ops
        fcm = tf.matmul(resh, kernel)
        output = tf.nn.bias_add(fcm, biases, name=name)

        if activation is not None:
            output = getattr(tf.nn, activation)(output, name=activation)
        if batch_norm:
            output = tf.nn.batch_normalization(output, mean=0, variance=1, offset=None,
                                scale=None, variance_epsilon=1e-8, name='batch_norm')
        if dropout is not None:
            output = tf.nn.dropout(output, dropout, seed=dropout_seed, name='dropout')
    return output

def initializer(kind='xavier', *args, **kwargs):
    if kind == 'xavier':
        init = tf.contrib.layers.xavier_initializer(*args, **kwargs)
    else:
        init = getattr(tf, kind + '_initializer')(*args, **kwargs)
    return init


def lrn(inp,
    depth_radius=5, 
    bias=1, 
    alpha=.0001, 
    beta=.75, 
    name = 'lrn',
    layer = None):
    with tf.variable_scope(layer):
        lrn = tf.nn.local_response_normalization(inp, depth_radius = depth_radius, alpha = alpha,
                                            beta = beta, bias = bias, name = name)
    return lrn

def conv(inp,
         out_depth,
         ksize=[3,3],
         strides=[1,1,1,1],
         padding='SAME',
         kernel_init='xavier',
         kernel_init_kwargs=None,
         bias=0,
         weight_decay=None,
         activation='relu',
         batch_norm=False,
         name='conv',
         layer = None,
         dilation = 1,
         deconv = False
         ):
    with tf.variable_scope(layer):
        # assert out_shape is not None
        if weight_decay is None:
            weight_decay = 0.
        if isinstance(ksize, int):
            ksize = [ksize, ksize]
            
        if isinstance(strides, int):
            strides = [1, strides, strides, 1]            
            
        if kernel_init_kwargs is None:
            kernel_init_kwargs = {}
        in_depth = inp.get_shape().as_list()[-1]

        # weights
        init = initializer(kernel_init, **kernel_init_kwargs)
        print 'deconv:', deconv 
        if deconv:
            kshape = [ksize[0], ksize[1], out_depth, in_depth]
            #kshape = [ksize[0], ksize[1], in_depth, out_depth]
        else:
            kshape = [ksize[0], ksize[1], in_depth, out_depth]


        kernel = tf.get_variable(initializer=init,
                                shape=kshape,
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='weights')
        init = initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')
        # ops
        if dilation == 1 and not deconv:
            conv = tf.nn.conv2d(inp, kernel,
                                strides=strides,
                                padding=padding)
        elif deconv:
            print ' deconv'
            shp = inp.get_shape().as_list()
            out_shape = [shp[0], shp[1] * strides[1], shp[2] * strides[1], out_depth]
            conv = tf.nn.conv2d_transpose(inp, kernel, out_shape,
                                strides=strides,
                                padding=padding)
        else:
            print 'regular'
            conv = tf.nn.atrous_conv2d(inp, kernel,
                    dilation,
                    padding=padding)
        output = tf.nn.bias_add(conv, biases, name=name)

        if activation is not None:
            output = getattr(tf.nn, activation)(output, name=activation)
        if batch_norm:
            output = tf.nn.batch_normalization(output, mean=0, variance=1, offset=None,
                                scale=None, variance_epsilon=1e-8, name='batch_norm')
    return output, kernel

