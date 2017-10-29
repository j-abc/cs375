from model_switcher import *
import os
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils

if __name__ == '__main__':
    '''
    this thing gon run it all
    '''
    # extract inputs
    model_name   = 'VAE'
    data_name    = 'cifar10'
    loss_name    = 'default'
    exp_id       = 'testrun'
    run_now      = True

    # specify the model
    my_model   = model_switcher(model_name = model_name, 
                              data_name = data_name, 
                              loss_name = loss_name, 
                              exp_id = exp_id)

    # yes i know this is a little janky, it's late rn
    experiment = my_model.exp_fn(my_model, my_model.exp_id)
    params = experiment.setup_params()

    # let's check our parameters for sanity
    if run_now:
        base.get_params()
        base.train_from_params(**params)
    else:
        print(params)