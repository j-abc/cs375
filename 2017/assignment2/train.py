from model_switcher import *
import os
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
import dill
import sys

if __name__ == '__main__':
    '''
    this thing gon run it all
    '''
    print (sys.argv)
    # extract inputs
    model_name   = 'colorful_model'
    data_name    = 'imagenet'
    loss_name    = 'colorful_loss'
    exp_id       = 'experiment2'
    run_now      = 'False'
    
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
        base.get_params()
        base.train_from_params(**params)
    else:
        print(params)
        print(dill.pickles(params))
