from support.output import *
from support.data import *
from keras_extensions.train import *

import argparse

from keras.models import Sequential
from keras.layers import *
from keras import optimizers

from keras import backend as TF

import numpy as np


#################################################################
# Model Hyperparameters
#################################################################

epochs = 13
batch_size = 40
lr = 1e-4

hidden_layer_1_size = 600
hidden_layer_1_dropout = 0.3
num_classes = 10


# Adam optimizer BEST RUN (98.68%)
# epochs = 2
# batch_size = 40
# lr = 0.0005
# dropout = 0.3
# hidden_layer_1_size = 600


#################################################################
# Program Architecture
#################################################################

def prep_data():
	(xs_train, ys_train), (xs_test, ys_test), data_shape = mnist_data(flatten=False)
	xs_train = np.expand_dims(xs_train, -1)
	xs_test = np.expand_dims(xs_test, -1)
	return (xs_train, ys_train), (xs_test, ys_test), np.shape(xs_train)[1:]

def build_model(data_shape):
	model = Sequential()

	model.add(Conv2D(32, (6,6), padding='same', input_shape=(data_shape[0], data_shape[1], 1)))
	model.add(Activation('relu'))
	model.add(Dropout(hidden_layer_1_dropout))
	model.add(MaxPooling2D(padding='same'))

	model.add(Conv2D(64, (4,4), padding='same'))
	model.add(Activation('relu'))
	model.add(Dropout(hidden_layer_1_dropout))
	model.add(MaxPooling2D(padding='same'))

	model.add(Reshape((7 * 7 * 64,)))

	# model.add(Dense(hidden_layer_1_size))
	# model.add(Activation('relu'))
	model.add(Dropout(hidden_layer_1_dropout))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

	return model

def train(model, train_data, test_data):
	print('\n=== Training ===')
	testChecker = TestSetCallback(test_data)
	model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=batch_size)

def test(model, test_data):
	print('\n=== Testing ===')
	scores = model.evaluate(test_data[0], test_data[1])
	print(Colors.OKGREEN + Colors.BOLD + 'Test Accuracy: {:.2%}'.format(scores[1]) + Colors.ENDC)
	print(Colors.OKBLUE + 'epochs={} batch_size={} hidden={} opt=adam lr={}'.format(epochs, batch_size, hidden_layer_1_size, lr) + Colors.ENDC +'\n')

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', action='store_true', help='print model visualization')
	return parser.parse_args()

def main():
	args = get_args()

	train_data, test_data, data_size = prep_data()

	model = build_model(data_size)

	if args.p:
		model.summary()
		viz_model(model)
		exit(0)

	train(model, train_data, test_data)
	test(model, test_data)


if __name__ == '__main__':
	main()
