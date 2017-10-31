import numpy as np 
from color_utils import preprocess
from skimage import data

a = data.astronaut()
D = np.zeros((2, a.shape[0], a.shape[1], a.shape[2]))

D[0,:,:,:] = a
D[1,:,:,:] = a

data_l, gt_ab_313, prior_boost_nongray = preprocess(D)

print np.nan_to_num(gt_ab_313)[0,:,:,:]