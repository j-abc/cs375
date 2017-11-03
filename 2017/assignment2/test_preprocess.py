import numpy as np 
from color_utils import preprocess, decode, softmax
from skimage import data

a = data.astronaut()
b = data.immunohistochemistry()
D = np.zeros((2, a.shape[0], a.shape[1], a.shape[2]), dtype=np.uint8)

D[0,:,:,:] = a
D[1,:,:,:] = b

data_l, gt_ab_313, prior_boost_nongray = preprocess(D)

print np.stack([x for x in D]).shape

Dp = decode(data_l, gt_ab_313)

print Dp.shape


sm = softmax(gt_ab_313)

print np.nan_to_num(sm)[0,:,:,:].sum(axis=-1)
