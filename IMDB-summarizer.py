from support.output import *
from support.data import *
from keras_extensions.maxout import *
from keras_extensions.train import *

from keras.models import Sequential
from keras.layers import *
from keras import optimizers
from keras import backend as K

import numpy as np
import argparse


#################################################################
# Model Hyperparameters
#################################################################

epochs = 1
batch_size = 20
lr = 0.05

num_classes = 2
word_embedding_size = 32
sent_embedding_size = 64
uttr_embedding_size = 512

# Adam optimizer BEST RUN (97.28%)
# epochs = 5
# batch_size = 40
# lr = 0.0005
# hidden_layer_1_size = 50
# number of competing weights = 2


#################################################################
# Program Architecture
#################################################################

def build_model(data_shape, word_index_cache):

	
	# input_layer = Input(shape=data_shape, dtype='int32', name='input')

	# x = TimeDistributed(Embedding(len(word_index_cache), embedding_size))(main_input)

	# 
	# lstm_out = LSTM(32)(x)


	model = Sequential()

	# data shape : (num_sentences, num_words_per_sentence)
	
	model.add(Embedding(len(word_index_cache), word_embedding_size, input_shape=data_shape))

	# data shape : (num_sentences, num_words_per_sentence, word_embedding_size)
	
	# model.add(TimeDistributed(LSTM(sent_embedding_size, activation='relu')))

	model.add(Reshape((data_shape[0], data_shape[1] * word_embedding_size)))
	model.add(Dense(sent_embedding_size, activation='relu'))

	# data shape : (num_sentences, sent_embedding_size)

	# model.add(LSTM(uttr_embedding_size, activation='relu'))

	model.add(Reshape((data_shape[0] * sent_embedding_size,)))
	model.add(Dense(uttr_embedding_size, activation='relu'))

	# data shape : (uttr_embedding_size)

	model.add(Dense(num_classes, activation='softmax'))

	# Use sparse categorical crossentropy as our loss function, since we have
	# output : [probability] @shape(10,)
	# target : integer label (note: NOT one-hot vector)
	model.compile(loss='sparse_categorical_crossentropy', 
		# optimizer=optimizers.Adam(lr=lr),
		optimizer=optimizers.SGD(lr=lr), 
		metrics=['accuracy'])

	return model

def train(model, train_data, test_data):
	print('\n=== Training ===')
	# testChecker = TestSetCallback(test_data)
	model.fit(train_data[0], train_data[1], epochs=epochs, batch_size=batch_size)#, callbacks=[testChecker])

def test(model, test_data):
	print('\n=== Testing ===')
	scores = model.evaluate(test_data[0], test_data[1])
	print(Colors.OKGREEN + Colors.BOLD + 'Test Accuracy: {:.2%}'.format(scores[1]) + Colors.ENDC)
	print(Colors.OKBLUE + 'epochs={} batch_size={} opt=sgd lr={}'.format(epochs, batch_size, lr) + Colors.ENDC +'\n')

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', action='store_true', help='print model visualization')
	return parser.parse_args()

def main():
	args = get_args()

	if args.p:
		data_shape = (20, 25)
		index_cache = [0] * 12000
	else:
		train_data, test_data, data_shape, index_cache = imdb_data('support/imdb_data', standardize_num_sen=20, standardize_sen_len=25)

	model = build_model(data_shape, index_cache)

	if args.p:
		model.summary()
		# viz_model(model, name='sentiment_summarizer_net.png')
		exit(0)

	print('\a')
	train(model, train_data, test_data)
	test(model, test_data)


if __name__ == '__main__':
	main()




