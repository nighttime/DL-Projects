print('\n\n### BEGIN RUN ###')

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *
from keras import optimizers

from keras import backend as TF

import numpy as np

class colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#################################################################
# Data and Hyperparameters
#################################################################

debug = False
plot = False

# xs_train, xs_test : [uint8] @shape(sample_size, 28, 28)
# ys_train, ys_test : [uint8] @shape(sample_size,) @range[0,9]
(xs_train, ys_train), (xs_test, ys_test) = mnist.load_data()

data_shape = np.shape(xs_train)[1:]

# xs_train, xs_test : [float] @shape(sample_size, 784) @range[0,1.0]
xs_train = np.expand_dims(xs_train, -1).astype(float) / 255.0
xs_test = np.expand_dims(xs_test, -1).astype(float) / 255.0

epochs = 1
batch_size = 20
lr = 1e-4

hidden_layer_1_size = 600
hidden_layer_1_dropout = 0.3
num_classes = 10

if debug:
	xs_train = xs_train[:10000,:]
	ys_train = ys_train[:10000]
	epochs = 1
	batch_size = 20
	hidden_layer_1_dropout = 0


#################################################################
# Model
#################################################################

model = Sequential()

model.add(Conv2D(32, (7,7), padding='same', input_shape=(data_shape[0], data_shape[1], 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(padding='same'))

model.add(Conv2D(64, (4,4), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(padding='same'))

model.add(Reshape((7 * 7 * 64,)))

model.add(Dense(hidden_layer_1_size))
model.add(Activation('relu'))
model.add(Dropout(hidden_layer_1_dropout))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['accuracy'])

model.summary()

if plot:
	from keras.utils import plot_model
	plot_model(model, show_shapes=True, to_file='model.png')
	print('generated chart of the model')
	exit(0)

#################################################################
# Training Phase
#################################################################

print('\n=== Training ===')
print('training on ' + str(np.shape(xs_train)[0]) + ' samples')

model.fit(xs_train, ys_train, epochs=epochs, batch_size=batch_size)


#################################################################
# Testing Phase
#################################################################

print('\n=== Testing ===')
print('testing on ' + str(np.shape(xs_test)[0]) + ' samples')
scores = model.evaluate(xs_test, ys_test)
print(colors.OKGREEN + colors.BOLD + 'Test Accuracy: {:.2%}'.format(scores[1]) + colors.ENDC)
print(colors.OKBLUE + 'epochs={} batch_size={} hidden={} opt=adam lr={}'.format(epochs, batch_size, hidden_layer_1_size, lr) + colors.ENDC)
print()

# Adam optimizer BEST RUN (97.04%)
# epochs = 4
# batch_size = 8
# lr = 0.005
# dropout = 0.3
# hidden_layer_1_size = 450