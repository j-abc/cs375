import tensorflow as tf
from deepretina.metrics import cc

A = tf.constant([
	[1.,2.,3.],
	[3.,4.,5.]
	])
B = tf.constant([
	[1.5,2.,3.5],
	[3.,4.5,5.5]
	])

loss = tf.py_func(cc, [A, B], tf.float32)

with tf.Session() as sess:
	a = loss.eval()
	print a