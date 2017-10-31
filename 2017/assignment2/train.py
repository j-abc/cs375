from model_switcher import *
import os
import numpy as np
import tensorflow as tf
import json
import argparse
from tfutils import base

if __name__ == '__main__':
    '''
    this thing gon run it all
    '''
    parser = argparse.ArgumentParser(prog='train net')
    parser.add_argument('filename', help='JSON filename with the exp details')
    args = parser.parse_args()
    print args
    with open(args.filename) as f:
        print 'I got here'
        setup = json.load(f)
    # extract inputs
    model_name   = setup['model_name']
    data_name    = setup['data_name']
    loss_name    = setup['loss_name']
    exp_id       = setup['exp_id']
    run_now      = setup['run_now']
    
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
    # super hacky shit, sorry john
    if 'devices' in setup:
        params['model_params']['devices'] = devices

    # let's check our parameters for sanity
    if run_now:
        base.train_from_params(**params)
    else:
        print(params)
