from keras.engine.topology import Layer
from keras import backend as K

class Maxout(Layer):
    def __init__(self, units, num_competitors, **kwargs):
        self.c = num_competitors
        self.output_size = units
        super(Maxout, self).__init__(**kwargs)

    def build(self, input_shape):
        # defined only for inputs with shape (batch_size, vec)
        assert len(input_shape) == 2
        # Create a trainable weight variable for this layer.
        self.competing_weights = self.add_weight(name='maxweight',
            shape=(self.c, input_shape[1], self.output_size),
            initializer='uniform',
            trainable=True)

        self.competing_biases = self.add_weight(name='maxbias',
            shape=(self.c, self.output_size),
            initializer='uniform',
            trainable=True)

        super(Maxout, self).build(input_shape)

    def call(self, X):
        # X : [batch_size, input_size]
        # W : [c, input_size, output_size]
        # S, Sb : [batch_size, c, output_size]
        # --> [batch_size, output_size]
        S  = K.dot(X, self.competing_weights)
        Sb = K.bias_add(S, self.competing_biases)
        return K.max(Sb, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, self.output_size)
