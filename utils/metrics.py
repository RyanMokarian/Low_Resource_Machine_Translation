import sacrebleu
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from utils import utils


class BleuScore():
    """This class computes the BLEU Metric."""
    def __init__(self):
        self.total_score = 0
        self.total_num_examples = 0

    def update_state(self, y_true, y_pred, vocab):
        for i in range(len(y_true)):
            label_sentence = utils.generate_sentence(y_true[i].numpy().astype('int'), vocab)
            pred_sentence = utils.generate_sentence_from_probabilities(y_pred[i].numpy(), vocab)
            self.total_score += sacrebleu.sentence_bleu(pred_sentence, label_sentence, smooth_method='exp').score
            self.total_num_examples += 1

    def result(self):
        return self.total_score / self.total_num_examples

    def reset_states(self):
        self.total_score = 0
        self.total_num_examples = 0


class Perplexity():
    """Compute perplexity Metric per token"""
    def __init__(self):
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.total_loss = 0
        self.total_num_examples = 0

    def calculate_loss(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.cross_entropy(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        self.total_loss += K.sum(loss_)
        self.total_num_examples += K.sum(mask)

    def update_state(self, y_true, y_pred):
        self.calculate_loss(y_true, y_pred)

    def result(self):
        return K.exp(self.total_loss / self.total_num_examples)

    def reset_states(self):
        self.total_loss = 0
        self.total_num_examples = 0
