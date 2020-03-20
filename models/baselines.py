import tensorflow as tf
from tensorflow.keras import layers

class DummyModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super(DummyModel, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(32, activation=tf.nn.relu)
        self.dense2 = layers.Dense(vocab_size, activation=None)
        
    def call(self, inputs):
        x = self.dense1(self.flatten(inputs))
        return self.dense2(x)