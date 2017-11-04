import matplotlib
import models
import losses
import experiments

class model_switcher:
    '''
    Description:
        model switcher is a wrapper around models.py.
    Inputs:
        model_name: name of model within models.py
        data_name:  name of data source
    Stores:
        dbname:   as data_name
        collname: as model_name + loss_name
        layers:   as layer names associated with our model
                  for now this is hard coded
        model_fn: a reference to the model definition as imported from models.py
    Example:
        from model_switcher import *
        my_model = model_switcher(model_name = 'layers', data_name = 'cifar10')
        # now we can access dbname, collname, and layers from my_model
    '''
    
    def __init__(self, model_name = 'herpaderp', data_name = 'cifar10', loss_name = 'default', exp_id = '', test_prefix = ''):
        '''
        sets up parameters/def associated with a given model and dataset
        '''
        # actual variables
        self.data_name  = data_name
        self.model_name = model_name
        self.loss_name  = loss_name
        self.exp_id     = exp_id
        self.test_id    = test_prefix + exp_id
        
        # variables that we feed into train and test.py
        self.dbname     = data_name
        self.collname   = model_name + '_' + loss_name
        self.layers     = self._list_model_layers(model_name)
        self.model_fn   = self._get_model_fn(model_name)
        self.loss_fn    = self._get_loss_fn(loss_name, model_name)
        self.exp_fn     = self._get_exp_fn(data_name)
        
    def _list_model_layers(self, model_name):
        '''
        defines the layers associated with a given model
        '''
        layer_dict = {
            'herpaderp':['test', 'test','test'],
            'tiny_model': ['blah'],
            'colorful_model':[
                'conv4_3','conv8_3'
            ],
            'VAE':['oh', 'geez', 'why'],
            'shallow_bottle':['conv1','deconv1'],
            'pooled_shallow':['conv1','pool1','deconv1'],
            'bottle_model': ['conv' + str(i) for i in range(1,3+1)] + ['deconv'+ str(i) for i in range(1, 3+1)]
        }
        if model_name not in layer_dict.keys():
            raise Exception('Model layer names not specified')
        else:
            return layer_dict[self.model_name]
    
    def _get_exp_fn(self, data_name):
        '''
        queries experiments.py for data set up of interest
        '''
        if hasattr(experiments, data_name):
            return getattr(experiments, data_name)
        else:
            raise Exception('Data name not found in experiments.py')
        
    def _get_model_fn(self, model_name):
        '''
        queries models.py for the model of interest
        '''
        if hasattr(models, model_name):
            return getattr(models,model_name)
        else:
            raise Exception('Model name not found in models.py')

    def _get_loss_fn(self, loss_name, model_name):
        '''
        queries loss.py for the loss of interest.
        if loss_name is denoted as 'default', then 
        set to a default value
        '''
        # set loss name to default if necessary
        default_dict = {
            'autoencoder':'autoencoder_loss',
            'VAE':'vae_loss',
            'colorful_model': 'colorful_loss',
            'shallow_bottle':'autoencoder_loss',
            'pooled_shallow':'autoencoder_loss',
            'bottle_model':'autoencoder_loss'
        }
        if loss_name == 'default':
            loss_name = default_dict[model_name]

        if hasattr(losses, loss_name):
            return getattr(losses, loss_name)
        else:
            raise Exception('Loss name not found in losses.py')

# may make this a dictionary of dictionaries

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

