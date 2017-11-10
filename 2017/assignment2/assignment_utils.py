import os
import numpy as np
import tensorflow as tf
import pymongo as pm
import gridfs
import cPickle
import scipy.signal as signal
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook, trange

def get_neural_exp_data(coll, exp_id):
    q_val = {'exp_id' : exp_id, 'validation_results' : {'$exists' : True}}
    
    # create our data dictionary
    val_steps = coll.find(q_val, projection = ['validation_results'])
    val_keys = {val_steps[i]['validation_results'].keys()[0]: val_steps[i]['validation_results'] for i in range(val_steps.count())}
    val_keys = {ikey: item[ikey] for ikey, item in val_keys.iteritems()}
    val_vals = [[ikey] + ikey.split('_') for ikey in val_keys.keys()]
    ddict = {(ilist[3], int(ilist[2])):val_keys[ilist[0]]for ilist in val_vals}
    
    # get unique steps and var levels
    import pandas as pd
    key_df = pd.DataFrame(val_vals, columns = ['key', 'fill', 'step', 'v'])
    uni_steps = key_df.step.unique().astype(int)
    uni_var = key_df.v.unique()
    
    return ddict, uni_steps, uni_var

def p_get_coll(collname, dbname):
    # connect to database
    dbname = dbname
    port = 24444

    conn = pm.MongoClient(port = port)
    db = conn[dbname]
    coll = conn[dbname][collname + '.files']

    # print out saved experiments in collection coll
    return coll, db

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
def get_validation_data(exp_id, coll, num = 10):

    q_val = {'exp_id' : exp_id, 'validation_results' : {'$exists' : True}, 'validates' : {'$exists' : False}}

    val_steps = coll.find(q_val, projection = ['validation_results'])
    my_range = np.linspace(0, val_steps.count()-1, num).tolist()
    my_range = set([int(round(i)) for i in my_range])
    l2_loss = [val_steps[i]['validation_results']['valid0']['l2_loss'] for i in my_range]
    img_inputs = [val_steps[i]['validation_results']['valid0']['gt'] for i in my_range]
    img_prediction = [val_steps[i]['validation_results']['valid0']['pred'] for i in my_range]
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


    
def p_get_data_list(coll, collname, step, v, exp_id, idx = -1):
    #exp_id = 'experiment_1_%s_%s_%s'%(str(step), collname, v)

    def get_neural_validation_data(exp_id):
        q_val = {'exp_id' : exp_id, 'validation_results' : {'$exists' : True}, 'validates': {'$exists' : True}}
        val_steps = coll.find(q_val, projection = ['validation_results', 'validates', 'exp_id'])
        results = [val_steps[i] for i in range(val_steps.count())]
        for res in results:
            try:
                res['step'] = coll.find({'_id': res['validates']})[0]['step']
            except:
                res['step'] = -1
        print(len(results))
        return results
    
    
    validation_data = get_neural_validation_data(exp_id=exp_id)
    all_keys = validation_data[idx]['validation_results'].keys()
    data_all = {}
    for key in all_keys:
        print key
        data = validation_data[idx]['validation_results'][key]
        extraction_step = key.split('_')[1]
        data_all[extraction_step] = data
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
        my_key = 'rdm_%s' % layer
        if my_key in data:
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
                if 'result_summary' in data[k][category]:
                    categorization_results = data[k][category]['result_summary']
            except:
                raise Exception(category + " is not a real category in data...")
        else:
            k = 'categorization_%s' % layer
            if 'result_summary' in data[k]:
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
            print(data.keys())
            k = 'it_regression_' + layer
            print('step', step, 'layer', layer, 1 - data[k]['noise_corrected_multi_rsquared_loss'])
        
def plot_conv1_kernels(data):
    """
    Plots the 96 conv1 kernels in a 12 x 8 subplot grid.
    
    You will need to EDIT this function.
    """
    if 'conv1_kernel' in data:
        kernels = np.array(data['conv1_kernel'])
        x = 5
        y = 6
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

def linear_classification(data):
    
    # Must set actual autoencoders
    autoencoders = [[1,20,30,65], [3,50,70,80], [2,7,65,70], [2,30,60,70], [1,62,64,68], [1,33,44,66], 
                 [3,50,60,70], [9,60,75,86], [4,5,6,7]]
    autoencoder_legend = ['shallow_bottleneck_cifar10', 'pooled_bottleneck_cifar10', 
                          'sparse_shallow_bottleneck_cifar10', 'sparce_pooled_cifar10',
                        'deem_symmetric_cifar10', 'vae_cifar10', 'colorful_cifar10', 
                          'shallow_bottleneck_imagenet', 'colorful_imagenet']
    x = range(0, len(autoencoders[0]))

    for autoencoder in autoencoders:
        plt.plot(x, autoencoder)
        #plt.plot(len(autoencoder), autoencoder)

    plt.xlabel('Layer')
    plt.ylabel('Top-1 Class Accuracy (%)')
    plt.title("Linear Classification")
    plt.ylim([0,100])
    plt.legend(autoencoder_legend, loc='best')

    plt.show()
### END OF YOUR CODE
