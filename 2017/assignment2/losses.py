import os
import numpy as np
import tensorflow as tf
from color_utils import preprocess

def colorful_loss(inputs, outputs):
    # preprocess images
    data_l, gt_ab_313, prior_boost_nongray = preprocess(inputs)

    flat_pred = tf.reshape(outputs['pred'], [-1, 313])
    flat_gt_ab_313 = tf.reshape(gt_ab_313, [-1,313])
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(flat_conv8_313, flat_gt_ab_313))

def vae_loss(inputs, outputs):
    # extract vars
    x = outputs['x']
    y = outputs['y']
    z_log_sigma = outputs['z_log_sigma']
    z_mu = outputs['z_mu']

    # p(x|z)
    log_px_given_z = -tf.reduce_sum(
        x * tf.log(y + 1e-10) +
        (1 - x) * tf.log(1 - y + 1e-10), 1)

    # d_kl(q(z|x)||p(z))
    # Appendix B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
    return tf.reduce_mean(log_px_given_z + kl_div)

def autoencoder_loss(inputs, outputs):
    print("INPUTS")
    print(inputs)
    print("OUTPUTS")
    #print(outputs['pred'])
    return     tf.nn.l2_loss(outputs['images'] - outputs['pred'])

def john_testing_stuff(inputs, outputs):
    print("INPUTS")
    print(inputs)
    print("OUTPUTS")
    #print(outputs['pred'])
    return 'no'     #tf.nn.l2_loss(outputs['images'] - outputs['pred']) tf.cast(tf.reduce_prod(tf.shape(outputs['images'])), tf.float32)


def val_loss_wrapper(inputs, outputs, loss_fn):
    return {'l2_loss':loss_fn(inputs,outputs),
            'pred':outputs['pred'], # change names later...
            'gt':inputs['images']}  # change names later...