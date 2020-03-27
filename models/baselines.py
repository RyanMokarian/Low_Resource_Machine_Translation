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

    def call(self, batch, training=False):
        return self.model(batch['inputs'])

class LSTM(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size=100, embedding_dim=256):
        super(LSTM, self).__init__()
        self.embed = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.dense1 = tf.keras.layers.LSTM(units=hidden_size)
        self.pred = tf.keras.layers.Dense(2)

    def __call__(self, inputs):
        emb = self.embed(inputs)
        d1 = self.dense1(emb)
        outputs = self.pred(d1)
        return outputs