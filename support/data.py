from keras.datasets import mnist
from keras.datasets import imdb
from keras.preprocessing.sequence import *
import numpy as np
import re
import random
# from collections import defaultdict
# from itertools import chain
import pdb

def mnist_data(flatten=False):
	"""Retrieves the MNIST image data and preprocesses it"""

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

def _cleaned_lines(review):
	"""Tokenizes a given review. Separates sentences on .?! and tokenizes on word chars only"""
	sents = re.split(r'\.|\?|\!+', review)
	word_sents = [[w for w in re.findall(r'\w+', s)] for s in sents]
	cleaned = [[w.lower() for w in s] for s in word_sents if s]
	return word_sents

def _encode_reviews(sets):
	"""Accepts any number of data sets and encodes them. Returns the index scheme produced"""
	d = {'':0, 'UNK':1}
	i = 2

	def enc(w):
		nonlocal i, d
		if w in d:
			return d[w]
		else:
			if len(d) > 12000:
				return 1
			else:
				d[w] = i
				i += 1
				return i-1

	for data in sets:
		data[:] = [[[enc(w) for w in s] for s in r] for r in data]

	return d

def _gen_xs_and_ys(pos, neg):
	"""Combines positive and negative sets and produces training labels while shuffling data"""
	# pos, neg = list(pos), list(neg)
	xs = pos + neg
	ys = ([1] * len(pos)) + ([0] * len(neg))
	xs_and_ys = list(zip(xs, ys))
	random.shuffle(xs_and_ys)
	xs, ys = zip(*xs_and_ys)
	return list(xs), list(ys)

def imdb_data(data_folder, standardize_num_sen=0, standardize_sen_len=0):
	"""Retrieves the IMDB movie review data and preprocesses it"""

	def _get_contents(fname):
		"""Opens the given file and cleans+tokenizes each review"""
		with open(data_folder + '/' + fname) as file:
			return [_cleaned_lines(review) for review in file]

	train = _gen_xs_and_ys(_get_contents('train_pos'), _get_contents('train_neg'))
	test  = _gen_xs_and_ys(_get_contents('test_pos'), _get_contents('test_neg'))

	print('-- loaded data')

	index_cache = _encode_reviews([train[0], test[0]])

	print('-- encoded data')

	def std_seq(seq, std, ext=[0]):
		"""Pads or truncates a sequence according to the given threshold"""
		if len(seq) < std:
			seq.extend(ext * (std - len(seq)))
		elif len(seq) > std:
			del seq[std:]

	if standardize_num_sen:
		for data in [train, test]:
			for r in data[0]:
				std_seq(r, standardize_num_sen, ext=[[0]])

	if standardize_sen_len:
		for data in [train, test]:
			for r in data[0]:
				for s in r:
					std_seq(s, standardize_sen_len)

	print('-- standardized data')

	return train, test, (standardize_num_sen, standardize_sen_len), index_cache





