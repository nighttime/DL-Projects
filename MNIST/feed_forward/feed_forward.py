from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import *

import numpy as np


#################################################################
# Data and Hyperparameters
#################################################################

# xs_train, xs_test : [uint8] @shape(sample_size, 28, 28)
# ys_train, ys_test : [uint8] @shape(sample_size,) @data(0-9)
(xs_train, ys_train), (xs_test, ys_test) = mnist.load_data()



#################################################################
# Functions
#################################################################


#################################################################
# Model
#################################################################

