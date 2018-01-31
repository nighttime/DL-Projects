from support.output import *
from support.data import *
from keras_extensions.maxout import *

import argparse

from keras.models import Sequential
from keras.layers import *
from keras import optimizers

from keras import backend as K

import numpy as np


#################################################################
# Model Hyperparameters
#################################################################

epochs = 5
batch_size = 40
lr = 0.0005

hidden_layer_1_size = 50
num_competitors = 2

num_classes = 10


# Adam optimizer BEST RUN (97.28%)
# epochs = 5
# batch_size = 40
# lr = 0.0005
# hidden_layer_1_size = 50
# number of competing weights = 2


#################################################################
# Program Architecture
#################################################################

def build_model(data_shape):
	model = Sequential()

	model.add(Maxout(hidden_layer_1_size, num_competitors, input_shape=data_shape))

	# model.add(Maxout(hidden_layer_2_size, 2))

	# Add a final dense layer
	model.add(Dense(num_classes, activation='softmax'))

	# Use sparse categorical crossentropy as our loss function, since we have
	# output : [probability] @shape(10,)
	# target : integer label (note: NOT one-hot vector)
	model.compile(loss='sparse_categorical_crossentropy', 
		optimizer=optimizers.Adam(lr=lr), 
		metrics=['accuracy'])

	return model

def train(model, train_data):
	print('\n=== Training ===')
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

	train_data, test_data, data_shape = mnist_data(flatten=True)

	model = build_model(data_shape)

	if args.p:
		model.summary()
		viz_model(model, name='maxout_net.png')
		exit(0)

	train(model, train_data)
	test(model, test_data)


if __name__ == '__main__':
	main()




