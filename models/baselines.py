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


class RecurrentNet(tf.keras.Model):
    """Multi-Layer LSTM OR GRU Model with a generate function for language modeling"""
    
    def __init__(self, vocab_size, embed_dims=256, rnn_type = 'gru' ,num_layers = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dims= embed_dims
        self.fc1 = layers.Embedding(vocab_size, embed_dims)
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.rnns= []
        for i in range(num_layers):
            if self.rnn_type == 'lstm':
                self.rnns.append(layers.LSTM(embed_dims, return_state= True, return_sequences=True))
            else:
                self.rnns.append(layers.GRU(embed_dims, return_state=True, return_sequences=True))
        
        self.fc2 = layers.Dense(vocab_size) 
        
    def call(self,x, training=False):
        x = self.fc1(x)
        for i in range(self.num_layers):
            x = self.rnns[i](x,training=False)[0]
        x = self.fc2(x)
        return x
        
    def generate(self,x,state=None,req_len=15):
        """Generate sequences of required length"""
        if state == None:
            state = [None]*self.num_layers
        outputs = []
        for j in range(req_len):
            new_state = []
            x = self.fc1(x)
            for i in range(self.num_layers):
                if self.rnn_type == 'lstm':
                    x,h,c = self.rnns[i](x, initial_state=state[i], training=False)
                    new_state.append([h,c])
                else:
                    x,h = self.rnns[i](x, initial_state=state[i], training=False)
                    new_state.append(h)
            x = self.fc2(x)
            # taking argmax gets stuck in a loop, hence sample from this distribution
            temp = 0.75 # hyperpararmeter to control predictable (low) or surprising (high) text
            x = tf.random.categorical(x[0]/temp, num_samples=1)[-1][None]
            outputs.append(x.numpy().flatten()[0])
            state = new_state
        return outputs,new_state