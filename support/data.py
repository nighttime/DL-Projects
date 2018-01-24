from keras.datasets import mnist
import numpy as np

def mnist_data(flatten=False):
	# xs_train, xs_test : [uint8] @shape(sample_size, 28, 28)
	# ys_train, ys_test : [uint8] @shape(sample_size,) @range[0,9]
	(xs_train, ys_train), (xs_test, ys_test) = mnist.load_data()

	# xs_train, xs_test : [float] @shape(sample_size, 28, 28) @range[0,1.0]
	xs_train = xs_train.astype(float) / 255.0
	xs_test  = xs_test.astype(float)  / 255.0

	data_shape = np.shape(xs_train)[1:]

	if flatten:
		flat_data_shape = data_shape[0] * data_shape[1]
		xs_train = np.reshape(xs_train, [-1, flat_data_shape])
		xs_test = np.reshape(xs_test, [-1, flat_data_shape])
		data_shape = (flat_data_shape,)

	return (xs_train, ys_train), (xs_test, ys_test), data_shape
