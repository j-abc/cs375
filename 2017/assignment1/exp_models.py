"""
Please implement a standard AlexNet model here as defined in the paper
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Note: Although you will only have to edit a small fraction of the code at the
beginning of the assignment by filling in the blank spaces, you will need to
build on the completed starter code to fully complete the assignment,
We expect that you familiarize yourself with the codebase and learn how to
setup your own experiments taking the assignments as a basis. This code does
not cover all parts of the assignment and only provides a starting point. To
fully complete the assignment significant changes have to be made and new
functions need to be added after filling in the blanks. Also, for your projects
we won't give out any code and you will have to use what you have learned from
your assignments. So please always carefully read through the entire code and
try to understand it. If you have any questions about the code structure,
we will be happy to answer it.

Attention: All sections that need to be changed to complete the starter code
are marked with EDIT!
"""

import os
import numpy as np
import tensorflow as tf
import cv2
from skimage.filters import gabor_kernel

def v1_model(inputs, train = True, norm = True, **kwargs):
    """
    This model is loosely based on the one decribed in
    Pinto, N., Cox, D. D. & DiCarlo, J. J.
    Why is Real-World Visual Object Recognition Hard. PLoS Comput Biol (2008)
    """

    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    input_to_network = inputs['images']

    # layers
    # only conv layer
    # convert to greyscale
    inputs_to_network = tf.image.rgb_to_grayscale(input_to_network)
    
    # input local response normalization
    lrn_in = lrn(inputs_to_network, depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    
    # convolve with gabor filters
    ## get gabor kernels
    orientations = [(1. / 16 * 2.) * i for i in range(16)]
    frequencies = [2., 3., 4., 6., 11., 18.]
    ksize = 43
    fixed_kernels =  get_gabor_kernels(ksize, orientations, frequencies)
    fixed_kernels = fixed_kernels[:,:,:43]
    print(fixed_kernels.shape)
    print(fixed_kernels.shape)
    print(fixed_kernels.shape)
    print(fixed_kernels.shape)
    print(fixed_kernels.shape)
    
    ## convolve
    outputs['conv1'],outputs['conv1_kernel']  = conv(lrn_in, 96, 11, 1, 
        padding='VALID', 
        layer = 'conv1', 
        fixed_kernels = fixed_kernels,
        )
    # threshold and response saturation (x > 0 := 0, 0 < x < 1:= x, x> 1:= 1)
    # output local divisive normalization
    lrn1 = outputs['conv1']
    if norm:
        lrn1 = lrn(outputs['conv1'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    outputs['pool1'] = max_pool(lrn1, 3, 2, layer = 'pool1')
    
    outputs['fc2'] = fc(outputs['pool1'], 4096, dropout=dropout, bias=.1, layer = 'fc2')
    #outputs['fc3'] = fc(outputs['fc2'], 4096, dropout=dropout, bias=.1, layer = 'fc3')
    outputs['fc4'] = fc(outputs['fc2'],1000, activation=None, dropout=None, bias=0, layer = 'fc4')

    outputs['pred'] = outputs['fc4']
    return outputs, {}  


def tiny_model(inputs, train = True, norm = True, **kwargs):

    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    input_to_network = inputs['images']

    ### YOUR CODE HERE

    # set up all layer outputs
    # set up all layer outputs
    outputs['conv1'],outputs['conv1_kernel']  = conv(outputs['images'], 96, 11, 4, padding='VALID', layer = 'conv1')
    lrn1 = outputs['conv1']
    if norm:
        lrn1 = lrn(outputs['conv1'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    outputs['pool1'] = max_pool(lrn1, 3, 2, layer = 'pool1')
    outputs['conv2'], outputs['conv2_kernel'] = conv(outputs['pool1'], 256, 5, 1, layer = 'conv2')
    lrn2 = outputs['conv2']
    if norm:
        lrn2 = lrn(outputs['conv2'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv2')
        
    outputs['pool2'] = max_pool(lrn2, 3, 2, layer = 'pool2')
    outputs['fc6'] = fc(outputs['pool2'], 4096, dropout=dropout, bias=.1, layer = 'fc6')
    outputs['fc7'] = fc(outputs['fc6'], 4096, dropout=dropout, bias=.1, layer = 'fc7')
    outputs['fc8'] = fc(outputs['fc7'],1000, activation=None, dropout=None, bias=0, layer = 'fc8')

    outputs['pred'] = outputs['fc8']
    return outputs, {}     

def alexnet_model(inputs, train=True, norm=True, **kwargs):
    """
    AlexNet model definition as defined in the paper:
    https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

    You will need to EDIT this function. Please put your AlexNet implementation here.

    Note:
    1.) inputs['images'] is a [BATCH_SIZE x HEIGHT x WIDTH x CHANNELS] array coming
    from the data provider.
    2.) You will need to return 'output' which is a dictionary where
    - output['pred'] is set to the output of your model
    - output['conv1'] is set to the output of the conv1 layer
    - output['conv1_kernel'] is set to conv1 kernels
    - output['conv2'] is set to the output of the conv2 layer
    - output['conv2_kernel'] is set to conv2 kernels
    - and so on...
    The output dictionary should include the following keys for AlexNet:
    ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1',
     'pool2', 'pool5', 'fc6', 'fc7', 'fc8']
    as well as the respective ['*_kernel'] keys for the kernels
    3.) Set your variable scopes to the name of the respective layers, e.g.
        with tf.variable_scope('conv1'):
            outputs['conv1'] = ...
            outputs['pool1'] = ...
    and
        with tf.variable_scope('fc6'):
            outputs['fc6'] = ...
    and so on.
    4.) Use tf.get_variable() to create variables, while setting name='weights'
    for each kernel, and name='bias' for each bias for all conv and fc layers.
    For the pool layers name='pool'.

    These steps are necessary to correctly load the pretrained alexnet model
    from the database for the second part of the assignment.
    """

    # propagate input targets
    outputs = inputs
    dropout = .5 if train else None
    input_to_network = inputs['images']

    ### YOUR CODE HERE

    # set up all layer outputs
    outputs['conv1'],outputs['conv1_kernel']  = conv(outputs['images'], 96, 11, 4, padding='VALID', layer = 'conv1')
    lrn1 = outputs['conv1']
    if norm:
        lrn1 = lrn(outputs['conv1'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv1')
    outputs['pool1'] = max_pool(lrn1, 3, 2, layer = 'pool1')
    
    
    outputs['conv2'], outputs['conv2_kernel'] = conv(outputs['pool1'], 256, 5, 1, layer = 'conv2')
    lrn2 = outputs['conv2']
    if norm:
        lrn2 = lrn(outputs['conv2'], depth_radius=5, bias=1, alpha=.0001, beta=.75, layer='conv2')

    outputs['pool2'] = max_pool(lrn2, 3, 2, layer = 'pool2')
    outputs['conv3'],outputs['conv3_kernel'] = conv(outputs['pool2'], 384, 3, 1, layer = 'conv3')
    outputs['conv4'],outputs['conv4_kernel'] = conv(outputs['conv3'], 384, 3, 1, layer = 'conv4')
    outputs['conv5'],outputs['conv5_kernel'] = conv(outputs['conv4'], 256, 3, 1, layer = 'conv5')
    outputs['pool5'] = max_pool(outputs['conv5'], 3, 2, layer = 'pool5')

    outputs['fc6'] = fc(outputs['pool5'], 4096, dropout=dropout, bias=.1, layer = 'fc6')
    outputs['fc7'] = fc(outputs['fc6'],4096, dropout=dropout, bias=.1, layer = 'fc7')
    outputs['fc8'] = fc(outputs['fc7'],1000, activation=None, dropout=None, bias=0, layer = 'fc8')

    outputs['pred'] = outputs['fc8']
    
    # provide access to kernels themselves
    #for key, value in outputs.iteritems():
    #    outputs[key + '_kernel'] = tf.get_default_graph().get_tensor_by_name('model_0/' + )

    # kernel = tf.get_variable(initializer=init,
    #                         shape=[ksize[0], ksize[1], in_depth, out_depth],
    #                         dtype=tf.float32,
    #                         regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
    #                         name='weights')
    ### END OF YOUR CODE

    for k in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'pool1',
            'pool2', 'pool5', 'fc6', 'fc7', 'fc8', 'conv1_kernel', 'pred']:
        assert k in outputs, '%s was not found in outputs' % k
    return outputs, {}
        #    return outputs['pred'], {}

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
         fixed_kernels = None,
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

        # has the option of using a fixed set of kernels
        if fixed_kernels.any():
            ksize = fixed_kernels.shape[:-1]
            out_depth = fixed_kernels.shape[0]
            kernel = tf.reshape(
                fixed_kernels, 
                (ksize[0], ksize[1], in_depth, out_depth)
                )
        else:
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

def get_gabor_kernels(ksize, orientations, frequencies):
    # [filter_height, filter_width, in_channels, out_channels]
    out_depth = len(orientations) * len(frequencies)
    kernels = np.zeros((ksize, ksize,out_depth), dtype=np.float32)
    ix = 0
    for i,orient in enumerate(orientations):
        for j,freq in enumerate(frequencies):
            kernels[:,:,ix]= cv2.getGaborKernel(
                (43, 43),
                9.0, 
                orient,
                freq,
                1,
                0,
                ktype=cv2.CV_32F
            )
            ix += 1
    return kernels
