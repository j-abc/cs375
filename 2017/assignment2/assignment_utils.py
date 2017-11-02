import os
import numpy as np
import tensorflow as tf
import pymongo as pm
import gridfs
import cPickle
import scipy.signal as signal
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, trange

def p_get_coll(collname, dbname):
    # connect to database
    dbname = dbname
    port = 24444

    conn = pm.MongoClient(port = port)
    coll = conn[dbname][collname + '.files']

    # print out saved experiments in collection coll
    return coll

def p22_training(exp_id, coll):
    def get_losses(exp_id):
        """
        Gets all loss entries from the database and concatenates them into a vector
        """
        q_train = {'exp_id' : exp_id, 'train_results' : {'$exists' : True}}
        return np.array([_r['loss'] 
                         for r in coll.find(q_train, projection = ['train_results']) 
                         for _r in r['train_results']])

    def get_boxcar(num_pts, total_length):
        my_ones = np.ones(num_pts)/num_pts
        return np.hstack([my_ones])

    ydict = {}
    ydict['losses'] = get_losses(exp_id = exp_id)
    smoothing_num = 100;
    my_box = get_boxcar(smoothing_num, ydict['losses'].shape[0])
    ydict['smooth' + str(smoothing_num)] = signal.convolve(ydict['losses'],my_box)

    plt.plot(ydict['losses'], color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(exp_id + " Training Loss vs. Iteration")
    plt.grid()
    plt.show()
    
    
    plt.plot(ydict['smooth' + str(smoothing_num)], color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(exp_id + " Smoothed Training Loss vs. Iteration")
    plt.grid()
    plt.show()
    
# Gets the validation data from the database
def get_validation_data(exp_id, coll):

    q_val = {'exp_id' : exp_id, 'validation_results' : {'$exists' : True}, 'validates' : {'$exists' : False}}
    val_steps = coll.find(q_val, projection = ['validation_results'])
    l2_loss = [val_steps[i]['validation_results']['valid0']['l2_loss'] 
        for i in range(0, val_steps.count(), 10)]
    img_inputs = [val_steps[i]['validation_results']['valid0']['gt'] 
        for i in range(0, val_steps.count(), 10)]
    img_prediction = [val_steps[i]['validation_results']['valid0']['pred']
        for i in range(0, val_steps.count(), 10)]
    return l2_loss, img_inputs, img_prediction

def plot_l2_loss(l2_loss, exp_id):

# We have provided a function that pulls the necessary data from the database. Your task is to plot the validation curve of the top1 and top5 accuracy. Label the graphs respectively and describe what you see.    
    ### PLOT VALIDATION RESULTS HERE
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(exp_id + " l2_loss: Loss vs. Step")
    plt.grid()
    ax1.plot(l2_loss)
    
    plt.show()

def p_get_data_list(coll, collname, step, v, idx = -1):
    exp_id = 'experiment_1_%s_%s_%s'%(str(step), collname, v)
    def get_neural_validation_data(exp_id):
        q_val = {'exp_id' : exp_id, 'validation_results' : {'$exists' : True}, 'validates': {'$exists' : True}}
        val_steps = coll.find(q_val, projection = ['validation_results', 'validates', 'exp_id'])

        results = [val_steps[i] for i in range(val_steps.count())]
        for res in results:
            try:
                res['step'] = coll.find({'_id': res['validates']})[0]['step']
            except:
                res['step'] = -1
        return results

    validation_data = get_neural_validation_data(exp_id=exp_id)
    data = validation_data[idx]['validation_results']['valid0']
    print validation_data[idx]['step'], v
    return data

def plot_rdms(data, target_layers, step):
    """
    Displays the RDMs of all target layers.

    You will need to EDIT this part. Please implement a function that visualizes 
    the rdms of all target_layers and displays the layer name as well as the 
    spearman_corrcoef in the plot title.


Please implement 'plot_rdms' that visualizes the rdms of all target_layers and displays the layer name as well as the spearman_corrcoef and the iteration step in the plot title.    
    """
    for i, layer in enumerate(target_layers):
        rdm = data['rdm_%s' % layer]
        spearman_corrcoef = data['spearman_corrcoef_%s' % layer]
        ### YOUR CODE HERE
        fig = plt.figure(figsize=(8, 8))
        m = fig.gca().matshow(rdm)
        plt.colorbar(m)
        plt.title(str(layer) + ": spearman_corr_coeff = " + str(spearman_corrcoef) + "; step: " + str(step))

        ### END OF YOUR CODE
        
def plot_categorization_results(data, target_layers, step, category=None):
    """
    Plots the confusion matrix and the average classification accuracy for each layer.

    You will need to EDIT this section.
    """
    for i, layer in enumerate(target_layers):
        if category:
            k = 'within_categorization_%s' % layer                
            try:
                categorization_results = data[k][category]['result_summary']
            except:
                raise Exception(category + " is not a real category in data...")
        else:
            k = 'categorization_%s' % layer
            categorization_results = data[k]['result_summary']
        ### YOUR CODE HERE
        fig = plt.figure(figsize=(8, 8))
        m = fig.gca().matshow(np.array(categorization_results['cms']).mean(2))
        plt.xticks(range(8), categorization_results['labelset'])
        plt.yticks(range(8), categorization_results['labelset'])
        plt.colorbar(m)
        ave_acc = (np.mean(categorization_results['accbal']) - .5) * 2
        if category:
            plt.title("category: " + category + " , " + k + "\n: mean_accuracy = " + str(ave_acc) + ", iteration_step = " + str(step))
        else:
            plt.title(k + ": mean_accuracy = " + str(ave_acc) + "\n, iteration_step = " + str(step))

        m.set_cmap('jet')
        ax = plt.gca()
        ax.xaxis.tick_bottom()
        
        
        ### END OF YOUR CODE
        
def plot_regression_results(data, target_layers, step):
    """
    Prints out the noise corrected multi rsquared loss for each layer.
    You will need to EDIT this function.
    """
    for layer in target_layers:
        try:
            k = 'it_regression_%s' % layer
            regression_results = data[k]
            ### YOUR CODE HERE
            calculated_regression_val = 1 - regression_results['noise_corrected_multi_rsquared_loss']
            print('step: ' + str(step) + ", layer: " + layer + ", calculated_value: " + str(calculated_regression_val))
        except:
            print 'Oh no...' + k + ' did not regress'
            pass
        ### END OF YOUR CODE
        
def plot_conv1_kernels(data, step):
    """
    Plots the 96 conv1 kernels in a 12 x 8 subplot grid.
    
    You will need to EDIT this function.
    """
    print('Iteration step: %d' % step)
    kernels = np.array(data['conv1_kernel'])
    ### YOUR CODE HERE
    x = 12
    y = 8
    f, axarr = plt.subplots(x, y, figsize=(4*x, 4*y))
    for i in range(x):
        for j in range(y):
            m = axarr[i,j].imshow(kernels[:,:,:,i*y + j])
            axarr[i,j].get_xaxis().set_visible(False)
            axarr[i,j].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0)
    
def plot_conv1_kernels2(data, step, x = 8, y = 5):
    """
    Plots the 96 conv1 kernels in a 12 x 8 subplot grid.
    
    You will need to EDIT this function.
    """
    print('Iteration step: %d' % step)
    kernels = np.array(data['conv1_kernel'])
    ### YOUR CODE HERE
    f, axarr = plt.subplots(x, y, figsize=(4*x, 4*y))
    for i in range(x):
        for j in range(y):
            m = axarr[i,j].imshow(kernels[:,:,0,i*y + j])
            axarr[i,j].get_xaxis().set_visible(False)
            axarr[i,j].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0)

def plot_images(data, step, x = 8, y = 2):
    ### YOUR CODE HERE
    f, axarr = plt.subplots(x, y)
    for i in range(x):
        for j in range(y):
            m = axarr[i,j].imshow(data[:,:,:,i*y + j])
            axarr[i,j].get_xaxis().set_visible(False)
            axarr[i,j].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=1)
    f.subplots_adjust(wspace=0)
    f.show()

### END OF YOUR CODE
