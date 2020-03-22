import tensorflow as tf
from tensorflow.keras import layers

class GRU(tf.keras.Model):
    def __init__(self, vocab_size, batch_size, rnn_units=3, embedding_dim=256):
        super(GRU, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                    batch_input_shape=[batch_size, None]),
            tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                stateful=True,
                                recurrent_initializer='glorot_uniform'),
            tf.keras.layers.Dense(vocab_size, activation=tf.nn.log_softmax)
        ])

    def call(self, inputs):
        return self.model(inputs)

