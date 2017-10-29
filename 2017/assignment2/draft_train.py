from model_switcher import *
from Experiments import * 

# honestly, i might just pop train and test into the same framework 

if __name__ == '__main__':
    # define inputs from command line
    model_name = sys.argv[0]
    data_name  = sys.argv[1]
    exp_id     = sys.argv[2]
    train_now  = {'True': True, 'False: 'False}[sys.argv[3]]
    
    # in train.py we specify which model to run and on which data set
#    model_name = 'tiny_model'
#    data_name = 'cifar10'
#    exp_id = 'experiment_1'
#    train_now = False

    # which model and dataset are we interested in? 
    my_model = model_switcher(model_name, data_name)

    # which dataset are we doing our experiment on?
    data_dict = {
        'cifar10': CIFAR10Experiment, 
        'imagenet': ImageNetExperiment
    }
    
    # get our experiment params: feed data set spec, feed model spec
    m = data_dict[data_name](my_model, exp_id)
    
    # now train it!
    if train_now:
        base.get_params()
        base.train_from_params(**m.params)