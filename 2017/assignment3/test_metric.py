import tensorflow as tf
from deepretina.metrics import cc
import numpy as np

A = tf.constant([
	[1.,2.,3.],
	[3.,4.,5.]
	])
B = tf.constant([
	[1.5,2.,3.5],
	[3.,4.5,5.5]
	])

D = np.array([
	[1.,2.,3.],
	[3.,4.,5.]
	])
E = np.array([
	[1.5,2.,3.5],
	[3.,4.5,5.5]
	])

print cc(D,E)

mean_loss, loss = tf.py_func(cc, [A, B], tf.float32)


with tf.Session() as sess:
	a = loss.eval()
	print a