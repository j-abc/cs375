import models
class model_switcher:
    '''
    Description:
        model switcher is a wrapper around models.py.
    Inputs:
        model_name: name of model within models.py
        data_name:  name of data source
    Stores:
        dbname:   as data_name
        collname: as model_name
        layers:   as layer names associated with our model
                  for now this is hard coded
        model_fn: a reference to the model definition as imported from models.py
    Example:
        from model_switcher import *
        my_model = model_switcher(model_name = 'layers', data_name = 'cifar10')
        # now we can access dbname, collname, and layers from my_model
    '''
    def __init__(self, model_name = 'herpaderp', data_name = 'cifar10'):
        '''
        sets up parameters/def associated with a given model and dataset
        '''
        # actual variables
        self.data_name  = data_name
        self.model_name = model_name
        
        # variables that we feed into train and test.py
        self.dbname     = data_name
        self.collname   = model_name
        self.layers     = self._model_layers(model_name)
        self.model_fn   = self._get_model_fn(model_name)
        
    def _get_model_fn(self, model_name):
        '''
        queries models.py for the model of interest
        '''
        if hasattr(models, model_name):
            return getattr(models,model_name)
        else:
            raise Exception('Model name not found in models.py')
            
    def _model_layers(self, model_name):
        '''
        defines the layers associated with a given model
        '''
        
        layer_dict = {
            'herpaderp':['test', 'test','test'],
            'tiny_model': ['blah']
        }

        if model_name not in layer_dict.keys():
            raise Exception('Model layer names not specified')
        else:
            return layer_dict[self.model_name]