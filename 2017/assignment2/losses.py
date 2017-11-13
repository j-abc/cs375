import os
import numpy as np
import tensorflow as tf
from color_utils import preprocess

def colorful_loss(inputs, outputs, **target_params):
    flat_pred = tf.reshape(outputs['conv8_313'], [-1, 313])
    flat_gt_ab_313 = tf.reshape(outputs['gt_ab_313'], [-1,313])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=flat_pred, labels=flat_gt_ab_313)
    return loss

def vae_loss(inputs, outputs, **target_params):
    # extract vars
    x = tf.contrib.layers.flatten(outputs['images'])
    y = tf.contrib.layers.flatten (outputs['pred'])
    z_log_sigma = outputs['z_logstd']
    z_mu = outputs['z_mean']

    # p(x|z)
    log_px_given_z = tf.reduce_sum(
        x * tf.log(y + 1e-10) +
        (1 - x) * tf.log(1 - y + 1e-10), 1)

    # d_kl(q(z|x)||p(z))
    # Appendix B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
    return - tf.reduce_mean(log_px_given_z - kl_div)

def autoencoder_loss(inputs, outputs):
    x_tensor = outputs['x_tensor']
    y = outputs['y']
    return tf.reduce_sum(tf.square(y - x_tensor))




def val_loss_wrapper(inputs, outputs, loss_fn):
    return {'l2_loss':loss_fn(inputs,outputs),
            'pred':outputs['pred'][::4], # change names later...
            'gt':inputs['images'][::4],
            'reconstruct':outputs['reconstruct'][::4],
            }  # change names later...}
