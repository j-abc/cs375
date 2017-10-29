import os
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
from dataprovider import CIFAR10DataProvider

class Experiment():
    def __init__(self, my_model):
        self.params = setup_params(self)

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
                'func': CIFAR10DataProvider,
                'data_path': self.Config.data_path,
                'group': 'train',
                'crop_size': self.Config.crop_size,
                # TFRecords (super class) data provider arguments
                'file_pattern': 'train*.tfrecords',
                'batch_size': self.Config.batch_size,
                'shuffle': False,
                'shuffle_seed': self.Config.seed,
                'n_threads': 4,
            },
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
                    'func': CIFAR10DataProvider,
                    'data_path': self.Config.data_path,
                    'group': 'val',
                    'crop_size': self.Config.crop_size,
                    # TFRecords (super class) data provider arguments
                    'file_pattern': 'test*.tfrecords',
                    'batch_size': self.Config.batch_size,
                    'shuffle': False,
                    'shuffle_seed': self.Config.seed,
                    'n_threads': 4,
                },
            }
        }

        """
        model_params defines the model i.e. the architecture that 
        takes the output of the data provider as input and outputs 
        the prediction of the model.
        """
        params['model_params'] = {
            'func': MY_MODEL # SEE THIS
        }

        """
        loss_params defines your training loss.
        """
        params['loss_params'] = {
        }

        """
        learning_rate_params defines the learning rate, decay and learning function.
        """
        params['learning_rate_params'] = {
            
        }

        """
        optimizer_params defines the optimizer.
        """
        params['optimizer_params'] = {
        }

        """
        save_params defines how, where and when your training results are saved
        in the database.
        """
        params['save_params'] = {
        }

        """
        load_params defines how and if a model should be restored from the database.
        """
        params['load_params'] = {
        }

        return params        

class CIFAR10Experiment():
    class Config():
        """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters. 
        """
        batch_size = 256
        data_path = '/datasets/cifar10/tfrecords'
        seed = 6
        crop_size = 24
        thres_loss = 1000000000000000
        n_epochs = 60
        train_steps = CIFAR10DataProvider.N_TRAIN / batch_size * n_epochs
        val_steps = np.ceil(CIFAR10DataProvider.N_VAL / batch_size).astype(int)    
    pass

class ImageNetExperiment():
    class Config():
        """
        Holds model hyperparams and data information.
        The config class is used to store various hyperparameters and dataset
        information parameters. 
        """
        batch_size = 256
        data_path = '/datasets/cifar10/tfrecords'
        seed = 6
        crop_size = 24
        thres_loss = 1000000000000000
        n_epochs = 60
        train_steps = CIFAR10DataProvider.N_TRAIN / batch_size * n_epochs
        val_steps = np.ceil(CIFAR10DataProvider.N_VAL / batch_size).astype(int)    
    pass


# Experiments on its own doesn't provide config
# we modify the parameters as we see fit...


# what do we switch here...
# shallow bottleneck convolutional autoencoder:
    # loss: L2 reconsturciton distance between original image nad predicted output
# pooled version: 
    # loss: see above
# sparse versions of 1 & 2
    # loss: terms for reconstruction loss and sparsity of activations in the hidden layer
# deep symmetric convolutional autoencoder
    # CAN explore if we should impose a sparseness penalty on the activations of the top encoder output
# VAE
    # in class (see loss function)
    # colorful image colorization network...
    # so basically... i should define a loss function for each model...
    # from a pool of losses
# so... hold losses in model_switcher.py
    # we can have default losses but also switch the losses if desired
