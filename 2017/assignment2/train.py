from model_switcher import *
import os
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
import sys

if __name__ == '__main__':
    '''
    this thing gon run it all
    '''
    if len(sys.argv) < 5:
        raise Exception("Not enough arguments! Requires: model_name, data_name, loss_name, and exp_id.")
    # extract inputs
    model_name   = sys.argv[1]  #'shallow_bottle'
    data_name    = sys.argv[2]  #'cifar10'
    loss_name    = sys.argv[3]  #'default'
    exp_id       = sys.argv[4]  #'testrun'
    run_now      = 'True'
    
    if run_now == 'True':
        run_now = True
    else:
        run_now = False

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
        sys.argv = [sys.argv[0]]
        base.get_params()
        print(params)
        base.train_from_params(**params)
    else:
        print(params)
       