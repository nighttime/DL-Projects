from keras import backend as K
from keras.callbacks import *
# from ..support.output import *

class TestSetCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        super().__init__()

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print(('\nTesting loss: {}, acc: {}\n').format(loss, acc))
        # print((Colors.BOLD + '\nTesting loss: {}, acc: {}\n' + Colors.ENDC).format(loss, acc))
        