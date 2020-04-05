#
# Most of the code comes from a tensorflow tutorial (https://www.tensorflow.org/tutorials/text/nmt_with_attention)
# and was adapted for this project.
#
import tensorflow as tf


class Seq2SeqGRU(tf.keras.Model):
    def __init__(self, vocab_size_en, vocab_fr, batch_size, config, embedding_matrix=None):
        super(Seq2SeqGRU, self).__init__()
        self.vocab_size_en = vocab_size_en
        self.vocab_fr = vocab_fr

        # FIXME : We should always take the the embedding_dim of the config.
        if embedding_matrix: 
            self.embedding_dim = embedding_matrix.shape[1]
        else:
            self.embedding_dim = config['embedding_dim']

        self.encoder_units = config['encoder_units']
        self.decoder_units = config['decoder_units']
        self.n_layers = config['n_layers']
        self.encoder = Encoder(self.vocab_size_en, self.embedding_dim, self.encoder_units, self.n_layers,
                               embedding_matrix)
        self.decoder = Decoder(len(self.vocab_fr), self.embedding_dim, self.decoder_units, self.n_layers)

    def call(self, batch, training=False):
        # Unpack inputs
        inputs, gen_seq_len = batch['inputs'], batch['gen_seq_len']
        batch_size = inputs.shape[0]  # infer batch_size on run time
        targets = batch['labels'] if training else None

        encoder_output, encoder_hidden = self.encoder(inputs)

        decoder_hidden = encoder_hidden
        decoder_input = tf.expand_dims([self.vocab_fr['<start>']] * batch_size, 1)

        predicted_seq = []
        predicted_seq.append(tf.one_hot([self.vocab_fr['<start>']] * batch_size, len(self.vocab_fr),
                                        dtype=tf.float32))  # <start> is the first prediction
        for t in range(1, gen_seq_len):
            # initialize decoder hidden state with zeros
            if t == 1:
                predictions, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output, None)
            else:  # carry over decoder hidden state
                predictions, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output,
                                                              decoder_hidden)
            predicted_seq.append(predictions)

            # using teacher forcing if training
            decoder_input = tf.expand_dims(targets[:, t], 1) if training else tf.expand_dims(
                tf.math.argmax(predictions, axis=1), 1)

        return tf.stack(predicted_seq, axis=1)

    def get_name(self):
        name = self.__class__.__name__
        name += f'_vocab-en_{self.vocab_size_en}'
        name += f'_vocab-fr_{len(self.vocab_fr)}'
        name += f'_embedding-dim_{self.embedding_dim}'
        name += f'_encoder-units_{self.encoder_units}'
        name += f'_decoder-units_{self.decoder_units}'
        name += f'_n-layers_{self.n_layers}'
        return name


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, encoder_units, n_layers, embedding_matrix):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weigths=[embedding_matrix])
        self.grus = [
            tf.keras.layers.GRU(encoder_units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform') for i in range(n_layers)
        ]

    def call(self, x):
        x = self.embedding(x)
        # None initalizes with zeros and takes care of batch_size
        for gru in self.grus:
            x, state = gru(x)
        # output, state = self.gru(x, initial_state=None)
        return x, state


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, n_layers):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.grus = [
            tf.keras.layers.GRU(decoder_units,
                                return_sequences=True,
                                return_state=True,
                                recurrent_initializer='glorot_uniform') for i in range(n_layers)
        ]
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = Attention(decoder_units)

    def call(self, x, hidden, enc_output, init_state):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        # let the decoder carry it's hidden state, change to None if not required
        for gru in self.grus:
            x, state = gru(x, initial_state=None)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights