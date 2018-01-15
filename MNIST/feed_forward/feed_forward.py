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
# Functions
#################################################################



#################################################################
# Data and Hyperparameters
#################################################################

debug = False

# xs_train, xs_test : [uint8] @shape(sample_size, 28, 28)
# ys_train, ys_test : [uint8] @shape(sample_size,) @range[0,9]
(xs_train, ys_train), (xs_test, ys_test) = mnist.load_data()

data_shape = np.shape(xs_train)[1:]
flat_img_size = data_shape[0] * data_shape[1]

# xs_train, xs_test : [float] @shape(sample_size, 784) @range[0,1.0]
xs_train = np.reshape(xs_train, [-1, flat_img_size]).astype(float) / 255.0
xs_test = np.reshape(xs_test, [-1, flat_img_size])

# Adam BEST RUN (97.04%)
# epochs = 4
# batch_size = 8
# lr = 0.005
# dropout = 0.3
# hidden = 450

epochs = 5
batch_size = 8
lr = 0.005

hidden_layer_1_size = 450
hidden_layer_1_dropout = 0.3
num_classes = 10

if debug:
	xs_train = xs_train[:10000,:]
	ys_train = ys_train[:10000]
	epochs = 1
	batch_size = 32
	hidden_layer_1_dropout = 0


#################################################################
# Model
#################################################################

model = Sequential()

# Add a basic dense layer with nonlinearity
model.add(Dense(hidden_layer_1_size, input_shape=(flat_img_size,), activation='sigmoid'))

# Add a minor dropout to weaken interdependence of neurons
model.add(Dropout(hidden_layer_1_dropout))

# Add a final dense layer
model.add(Dense(num_classes, activation='softmax'))

# Use sparse categorical crossentropy as our loss function, since we have
# output : [probability] @shape(10,)
# target : integer label (note: NOT one-hot vector)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(lr=0.005), metrics=['accuracy'])

model.summary()


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







