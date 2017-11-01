from model_switcher import *
import os
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
import pymongo as pm
import sys
from neuralexperiments import *

# Ramon's little helpers
def roundup(x, nearest=10000.):
    return int(math.ceil(x / nearest)) * int(nearest)


def get_relevant_steps(modelname, quantiles, my_model):
    # get connection
    port = 24444
    host = 'localhost'
    connection = pm.MongoClient(port = port, host = host)
    coll = connection[my_model.data_name][modelname]
    # obtain max steps
    query = {
        'step':{'$exists':True}, 
        'validates': {'$exists': False},
        'saved_filters': True,
        }
    step_query = coll.find(query,
              sort=[("step", pm.ASCENDING)],
              projection=['step'],
             )
    # get quantile steps
    steps = [step_query[i]['step'] for i in range(step_query.count())]
    indices = [int(round(q*len(steps) -1)) for q in quantiles]
    step_qs = [steps[i] for i in indices]
    return step_qs

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

    # sys.argv = [sys.argv[0]] # pop out for now
    """
    image_sets = [
        ['V0', 'V3', 'V6'],
        ['V6'],
    ]
    """
    image_sets = [['V0', 'V3', 'V6']]
    quantiles = [1]
    training_points = {
        my_model.collname: get_relevant_steps(my_model.collname + '.files', quantiles, my_model)
    }
    
    for image_set in image_sets:
        print 'Running ', my_model.collname, 'on these image sets: V6'
        for training_point in  training_points[my_model.collname]: 
            m = NeuralDataExperiment(my_model, extraction_step=training_point)
            params = m.setup_params()
            base.test_from_params(**params)