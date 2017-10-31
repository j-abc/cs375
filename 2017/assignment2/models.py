import os
import numpy as np
import tensorflow as tf
from color_utils import preprocess
from losses import *

'''
TODO: 
- change the loss to be _out_ of the model, the model should just return 
the values at each layer. Specify in a different function the loss functions
- merge autoencoder and var-autoencoder: just return a parameter if you want to make
variational
- add the colorization model
'''

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

        kernel = tf.get_variable(initializer=init,
                                shape=[ksize[0], ksize[1], in_depth, out_depth],
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
        conv = tf.nn.conv2d(inp, kernel,
                            strides=strides,
                            padding=padding)
        output = tf.nn.bias_add(conv, biases, name=name)

        if activation is not None:
            output = getattr(tf.nn, activation)(output, name=activation)
        if batch_norm:
            output = tf.nn.batch_normalization(output, mean=0, variance=1, offset=None,
                                scale=None, variance_epsilon=1e-8, name='batch_norm')
    return output, kernel

def deconv(inp, 
           out_shape, 
           ksize = [3,3],
           strides = [1,1,1,1],
           padding = 'SAME',
           kernel_init = 'xavier',
           kernel_init_kwargs = None,
           bias = 0, 
           weight_decay = None,
           batch_norm = False,
           name = 'deconv', 
           layer = None,
           dilation = 1):

    with tf.variable_scope(layer):
        # assert out_shape is not None
        if weight_decay is None:
            weight_decay = 0.
        if isinstance(ksize, int):
            ksize = [ksize, ksize]
#        if isinstance(strides, int):
        strides = [1, strides, strides, 1]  
        print("STRIDES: ")
        print(strides)
        if kernel_init_kwargs is None:
            kernel_init_kwargs = {}            
        in_depth = inp.get_shape().as_list()[-1]
        out_depth = out_shape[-1]
        
        input_channels = inp.get_shape()[3] # 
        # output_shape = [filter_size, filter_size, output_channels, input_channels] # kernel size # taken care of
        print out_shape
        batch_size = tf.shape(inp)[0]
        '''
        output_channels = out_shape[-1]
        input_height = tf.shape(inp)[1]
        input_width= tf.shape(inp)[2]
        
        out_rows = input_height*strides[1]
        out_cols = input_width*strides[1]
        
        deconv_out_shape = [batch_size, out_rows, out_cols, output_channels]
        '''
        # define convolutional kernel and bias
        init = initializer(kernel_init, **kernel_init_kwargs)        
        kernel = tf.get_variable(initializer=init,
                                shape=[ksize[0], ksize[1], out_depth , in_depth]        ,
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='weights')
        init = initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                shape=[out_depth],
                                dtype=tf.float32,
                                regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                name='bias')
                                
        
        # feed into our deconvolution
        deconv = tf.nn.conv2d_transpose(inp, kernel, out_shape,#deconv_out_shape, 
                                        strides = strides)
        
        deconv = tf.nn.bias_add(deconv, biases, name=name)
        
        output = deconv
        if batch_norm:
            output = tf.nn.batch_normalization(output, mean=0, variance=1, offset=None,
                                scale=None, variance_epsilon=1e-8, name='batch_norm')    
    return output, kernel

def shallow_bottle(inputs, train = True, norm = True, **kwargs):
    # propagate input targets
    outputs = inputs
    
    print inputs['images'].get_shape().as_list()
    print outputs['images'].get_shape().as_list()
    # https://stackoverflow.com/questions/39373230/what-does-tensorflows-conv2d-transpose-operation-do
    
    # create the encoder
    outputs['conv1'], outputs['conv1_kernel'] = conv(outputs['images'], 
                                                     64, 
                                                     ksize=7, 
                                                     strides=16, 
                                                     padding = 'SAME', 
                                                     layer = 'conv1',
                                                     activation = 'relu')
    
    # create the decoder
    my_shape =  outputs['images'].get_shape().as_list()
    out_shape = [my_shape[0], my_shape[1], my_shape[2], my_shape[3]]
    print("out shape")
    print out_shape
    print inputs['images'].get_shape().as_list()
    outputs['deconv1'], outputs['deconv1_kernel'] = deconv(outputs['conv1'], 
                                                           out_shape,
                                                           ksize=7,
                                                           strides=16,
                                                           padding = 'SAME',
                                                           layer = 'deconv1')
    # outputs['out'] 
    outputs['pred'] = outputs['deconv1']
    return outputs, {}
    
    
def stuff(inputs, train = True, norm = True, **kwargs):
    # propagate input targets
    outputs = inputs
    
    print inputs['images'].get_shape().as_list()
    print outputs['images'].get_shape().as_list()

    
    # create the encoder
    outputs['conv1'], outputs['conv1_kernel'] = conv(outputs['images'], 
                                                     64, 
                                                     ksize=7, 
                                                     strides=16, 
                                                     padding = 'VALID', 
                                                     layer = 'conv1',
                                                     activation = 'relu')
    
    # create the decoder
    my_shape =  outputs['images'].get_shape().as_list()
    out_shape = [my_shape[0], my_shape[1], my_shape[2], my_shape[3]]
    print("out shape")
    print out_shape
    print inputs['images'].get_shape().as_list()
    outputs['deconv1'], outputs['deconv1_kernel'] = deconv(outputs['conv1'], 
                                                           out_shape,
                                                           ksize=7,
                                                           strides=12,
                                                           padding = 'VALID',
                                                           layer = 'deconv1')
    # outputs['out'] 
    outputs['pred'] = outputs['deconv1']
    return outputs, {}

def pooled_shallow(inputs, train = True, norm = True, **kwargs):
    # propagate input targets
    outputs = inputs
    
    print inputs['images'].get_shape().as_list()
    print outputs['images'].get_shape().as_list()

    
    # create the encoder
    outputs['conv1'], outputs['conv1_kernel'] = conv(outputs['images'], 
                                                     64, 
                                                     ksize=7, 
                                                     strides=4, 
                                                     padding = 'VALID', 
                                                     layer = 'conv1',
                                                     activation = 'relu')
    outputs['pool1'] = max_pool(outputs['conv1'], 3, 4, layer = 'pool1')    
        
    # create the decoder
    my_shape =  outputs['images'].get_shape().as_list()
    out_shape = [my_shape[0], my_shape[1], my_shape[2], my_shape[3]]
    outputs['deconv1'], outputs['deconv1_kernel'] = deconv(outputs['pool1'], 
                                                           out_shape,
                                                           ksize=7,
                                                           strides=3,
                                                           padding = 'VALID',
                                                           layer = 'deconv1')
    # outputs['out'] 
    outputs['pred'] = outputs['deconv1']
    return outputs, {}