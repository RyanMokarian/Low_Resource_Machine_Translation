#
# Most of the code comes from a tensorflow tutorial (https://www.tensorflow.org/tutorials/text/nmt_with_attention)
# and was adapted for this project.
#

import tensorflow as tf

class Seq2SeqGRU(tf.keras.Model):
    def __init__(self, vocab_size_en, vocab_fr, batch_size, embedding_dim, encoder_units, decoder_units, attention_units):
        super(Seq2SeqGRU, self).__init__()
        self.batch_size = batch_size
        self.vocab_fr = vocab_fr
        self.encoder = Encoder(vocab_size_en, embedding_dim, encoder_units, batch_size)
        self.decoder = Decoder(len(vocab_fr), embedding_dim, decoder_units, batch_size)
   
    def call(self, batch, training=False):
        # Unpack inputs
        inputs, gen_seq_len = batch['inputs'], batch['gen_seq_len']
        targets = batch['labels'] if training else None

        encoder_output, encoder_hidden = self.encoder(inputs, self.encoder.initialize_hidden_state())

        decoder_hidden = encoder_hidden
        # TODO : Change the line below to this one -> decoder_input = tf.expand_dims([self.vocab_fr['<start>']] * self.batch_size, 1)
        decoder_input = tf.expand_dims([0] * self.batch_size, 1)
        predicted_seq = []
        for t in range(gen_seq_len):
            predictions, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_output)
            predicted_seq.append(predictions)

            # using teacher forcing if training
            decoder_input = tf.expand_dims(targets[:, t], 1) if training else tf.expand_dims(tf.math.argmax(predictions, axis=1), 1)
        
        return tf.stack(predicted_seq, axis=1)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_size, self.enc_units))

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, decoder_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.decoder_units = decoder_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.decoder_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = Attention(self.decoder_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

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
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights