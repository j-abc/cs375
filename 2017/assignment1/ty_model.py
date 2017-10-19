from skimage.filters import gabor_kernel
from skimage import io
from matplotlib import pyplot as plt  
import numpy as np


def single_frequency_bank(frq, bndwdth, offst, n_stds, n_steps): 

    tmp_filter_bank = []
    step_size = float(1) / float(n_steps)
    gabor_orientations = np.arange(0.0 + step_size, 1 + step_size, step_size) * np.pi
    filter_bank = np.zeros([11,11,len(gabor_orientations)])
    
    for i_filter in range(0, len(gabor_orientations)):
        
        tmp_gabor = gabor_kernel(frequency=frq, 
				bandwidth=bndwdth, 
				offset = offst, 
				theta = gabor_orientations[i_filter], 
				n_stds=.8)

        tmp_filter = tmp_gabor.real
        
        if np.shape(tmp_gabor.real) == (11, 11):           

            final_filter = tmp_gabor.real

        else: 

            onset = int(round((np.shape(tmp_gabor.real)[0] - 11)/2))
            offset = int(onset + 11)
            final_filter = tmp_gabor.real[onset:offset,onset:offset]

        filter_bank[:,:,i_filter] = final_filter  

    return filter_bank


def create_filterbanks():

    # prespecified parameters  
    freq = [.15, .1, .05, .5]
    offset = [0, 0, 1, 1.6]
    steps = [20, 20, 20, 20,]
    band = [.25, .5, 1, .01]
    n_stds = [1.1, .6, 1, 1.5]

    filters = []

    for fre, ban, off, std, ste in zip(freq, band, offset, n_stds, steps):
        filters.append(single_frequency_bank(fre, ban, off, std, ste)) 
    
    # not beautiful, but okay: need to reformat into array, preserving information
    filters = np.concatenate((filters[0], filters[1], filters[2], filters[3]), axis=2)
    # create three channels for each filter bank--r, g, and b
    filter_bank = np.stack((filters, filters, filters), axis=2)
	
    return filter_bank

def gabor_model(inputs, **kwargs):

    outputs = inputs
    
    def conv(input, kernel):
        conv = tf.nn.conv2d(input, kernel, strides=[1,1,1,1], padding='SAME')
        return conv

    
    gabors_ = generate_gabor_filterbank.create_filterbanks()
    gabors = tf.convert_to_tensor(gabors_, dtype=np.float32)

    images = outputs['images']
    conv_in = conv(images, gabors)

    outputs['conv_kernel'] = gabors
    outputs['conv'] = tf.nn.relu(conv_in)
    lrn = tf.nn.local_response_normalization(outputs['conv'],depth_radius = 5, alpha=.0001,beta=.75, bias=1)
    pool = tf.nn.max_pool(lrn, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    print('conv: ', gabors.get_shape())
    print('conv_kernel: ', outputs['conv_kernel'])
    return outputs, {}
