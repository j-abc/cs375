import os
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
from dataprovider import CIFAR10DataProvider, ImageNetDataProvider
from losses import *

class Experiment():
    def __init__(self, model, exp_id):
        self.model = model
        self.exp_id = exp_id
        
    def setup_params(self):
        """
        This function illustrates how to setup up the parameters for 
        train_from_params. 
        """
        params = {}

        """
        train_params defines the training parameters consisting of 
            - the data provider that reads the data, preprocesses it and enqueues it into
              the data queue
            - the data queue that batches and if specified shuffles the data and provides 
              the input to the model
            - other configuration parameters like the number of training steps
        It's arguments are
            data_params: defines how the data is read in.
            queue_params: defines how the data is presented to the model, i.e.
            if it is shuffled or not and how big of a batch size is used.
            targets: the targets to be extracted and evaluated in the tensorflow session
            num_steps: number of training steps
            thres_loss: if the loss exceeds thres_loss the training will be stopped
            validate_first: run validation before starting the training
        """
        params['train_params'] = {
            'data_params': {
                # Cifar 10 data provider arguments
                'func': self.Config.fnDataProvider,
                'data_path': self.Config.data_path,
                'group': 'train',
                'crop_size': self.Config.crop_size, # not in tut2
                # TFRecords (super class) data provider arguments
                'file_pattern': 'train*.tfrecords', # not in tut2
                'batch_size': self.Config.batch_size, # not in tut2
                'shuffle': False, # not in tut2
                'shuffle_seed': self.Config.seed, # not in tut2
                'n_threads': 4, 
            },
            'queue_params': {
                'queue_type': 'random',
                'batch_size': self.Config.batch_size,
                'seed': self.Config.seed, # not in tut2
                'capacity': self.Config.batch_size * 10, # not in tut2
                'min_after_dequeue': self.Config.batch_size * 5, #not in tut2
            },
            'num_steps': self.Config.train_steps,
            'thres_loss': self.Config.thres_loss,
            'validate_first': False,            
        }
        
        """
        validation_params similar to train_params defines the validation parameters.
        It has the same arguments as train_params and additionally
            agg_func: function that aggregates the validation results across batches,
                e.g. to calculate the mean of across batch losses
            online_agg_func: function that aggregates the validation results across
                batches in an online manner, e.g. to calculate the RUNNING mean across
                batch losses
        """
        params['validation_params'] = {
            'valid0': {
                'data_params': {
                    # Cifar 10 data provider arguments
                    'func': self.Config.fnDataProvider,
                    'data_path': self.Config.data_path, # not in tut2
                    'group': 'train', #'val' is what's in as1
                    'crop_size': self.Config.crop_size, # not in tut2
                    # TFRecords (super class) data provider arguments
                    'file_pattern': 'test*.tfrecords', #not in tut2
                    'batch_size': self.Config.batch_size, #not in tut2
                    'shuffle': False, #not in tut2
                    'shuffle_seed': self.Config.seed, #not in tut2
                    'n_threads': 4, #not in tut2
                },
               'queue_params': {
                    'queue_type': 'fifo',
                    'batch_size': self.Config.batch_size, 
                    'seed': self.Config.seed, # not in tut2
                    'capacity': self.Config.batch_size * 10, # not in tut2
                    'min_after_dequeue': self.Config.batch_size * 5, #not in tut2
                },
                '''
                'targets': {
                    'func': vae_loss, # using our loss function for validation
                },
                '''
                'targets': {
                    'func': self.model.loss_fn,
                    'target': ['labels'],
                    'loss_per_case_func_params' : {'_outputs': 'outputs', 
                        '_targets_$all': 'inputs'
                    },
                    'agg_func': tf.reduce_mean,
                },
                'num_steps': self.Config.val_steps,                
                'num_steps': self.Config.val_steps,
                'agg_func': self.agg_mean, 
                'online_agg_func': self.online_agg_mean,             
            }
        }
        """
        model_params defines the model i.e. the architecture that 
        takes the output of the data provider as input and outputs 
        the prediction of the model.
        """
        params['model_params'] = {
            'func': self.model.model_fn # ... if you wanna add model parameters in
                                        # please do through model_switcher
                                        # doing it here will muddy up file naming conventions for
                                        # dbname or collname or w/e it's called. we want to be consistent
        }

        """
        loss_params defines your training loss.
        """
        params['loss_params'] = {
            'targets': ['images'],
            'agg_func': tf.reduce_mean,
            'loss_per_case_func': self.model.loss_fn,
            'loss_per_case_func_params' : {'_outputs': 'outputs', 
                '_targets_$all': 'inputs'},
            'loss_func_kwargs' : {},
        }

        """
        learning_rate_params defines the learning rate, decay and learning function.
        """
        def piecewise_constant_wrapper(global_step, boundaries, values):
            return tf.train.piecewise_constant(global_step, boundaries, values)  

        params['skip_check'] = True
        '''
        # this params from as1
        params['learning_rate_params'] = {
            'func': piecewise_constant_wrapper,
            'boundaries': list(np.array([150000, 300000, 450000]).astype(np.int64)),
            'values': [0.01, 0.005, 0.001, 0.0005]            
        }
        '''
        
        # this params taken from the tutorial
        params['learning_rate_params'] = {
            'learning_rate': 5e-3,
            'decay_steps': 2000,
            'decay_rate': 0.95,
            'staircase': True,
        }

        """
        optimizer_params defines the optimizer.
        """
        # this params taken from the tutorial
        params['optimizer_params'] = {
            'func': optimizer.ClipOptimizer,
            'optimizer_class': tf.train.AdamOptimizer,
            'clip': False,
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.
        """
        params['save_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': self.model.dbname,
            'collname': self.model.collname,
            'exp_id': self.exp_id,
            'save_valid_freq': 200, # 10000, in as1
            'save_filters_freq': 1000, # 30000, in as1
            'cache_filters_freq': 1000, # 50000, in as1
            'save_metrics_freq': 200,
            'save_initial_filters' : False, # not in tutorial2, maybe take out
            'save_to_gfs': [], # not in tutorial2, maybe take out
        }

        """
        load_params defines how and if a model should be restored from the database.
        """
        params['load_params'] = {
            'host': 'localhost',
            'port': 24444,
            'dbname': self.model.dbname,
            'collname': self.model.collname,
            'exp_id': self.exp_id,
            'do_restore': True,
            'load_query': None,            
        }
        return params
    
    def online_agg_mean(self, agg_res, res, step):
        """
        Appends the mean value for each key
        """
        if agg_res is None:
            agg_res = {k: [] for k in res}
        for k, v in res.items():
            if k in ['pred', 'gt']:
                value = v
            else:
                value = np.mean(v)
            agg_res[k].append(value)
        return agg_res

    def agg_mean(self, results):
        for k in results:
            if k in ['pred', 'gt']:
                results[k] = results[k][0]
            elif k is 'l2_loss':
                results[k] = np.mean(results[k])
            else:
                raise KeyError('Unknown target')
        return results


    def subselect_tfrecords(self, path):
        """
        Illustrates how to subselect files for training or validation
        """
        all_filenames = os.listdir(path)
        rng = np.random.RandomState(seed=SEED)
        rng.shuffle(all_filenames)
        return [os.path.join(path, fn) for fn in all_filenames
                if fn.endswith('.tfrecords')]

class cifar10(Experiment):
    class Config():
        # provided [edit these]
        fnDataProvider = CIFAR10DataProvider
        batch_size = 256
        data_path = '/datasets/cifar10/tfrecords'
        seed = 6
        crop_size = 24
        thres_loss = 1000000000000000
        n_epochs = 60
        
        # calculated
        train_steps = fnDataProvider.N_TRAIN / batch_size * n_epochs
        val_steps = np.ceil(fnDataProvider.N_VAL / batch_size).astype(int)

class imagenet(Experiment):        
    class Config():
        """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters.
        Please set the seed to your group number. You can also change the batch
        size and n_epochs if you want but please do not change the rest.
        """
        # provided [edit these]
        fnDataProvider = ImageNetDataProvider
        batch_size = 256
        data_path = '/datasets/TFRecord_Imagenet_standard'
        seed = 6
        crop_size = 227 # also wut this
        thres_loss = 1000000000000000 # dafuq does this mean?
        n_epochs = 90
        
        # calculated
        train_steps = fnDataProvider.N_TRAIN / batch_size * n_epochs
        val_steps = np.ceil(fnDataProvider.N_VAL / batch_size).astype(int)        