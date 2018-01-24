from keras.engine.topology import Layer
from keras import backend as K

class DenseMaxout(Layer):
    def __init__(self, output_dim, num_competitors, **kwargs):
        self.output_dim = output_dim
        self.num_competitors = num_competitors
        super(DenseMaxout, self).__init__(**kwargs)

    def build(self, input_shape):
        # defined only for inputs with shape (batch_size, vec)
        assert len(input_shape) == 2
        # Create a trainable weight variable for this layer.
        self.competing_weights = self.add_weight(name='maxweight',
            shape=(self.num_competitors, input_shape[1], self.output_dim),
            initializer='uniform',
            trainable=True)

        self.competing_biases = self.add_weight(name='maxbias',
            shape=(self.num_competitors, self.output_dim,),
            initializer='uniform',
            trainable=True)

        super(DenseMaxout, self).build(input_shape)

    def call(self, x):
        # x : [batch_size, 768]
        # w : [k, 768, 500]
        # ->  [batch_size, k, 500]
        # --> [batch_size, 500]
        signals = K.dot(x, self.competing_weights)
        biased_signals = K.bias_add(signals, self.competing_biases)
        return K.max(biased_signals, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim)
