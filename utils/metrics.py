
import sacrebleu
import numpy as np
import tensorflow as tf

from utils import utils

class BleuScore():

    def __init__(self):
        self.true_sentences = []
        self.pred_sentences = []

    def update_state(self, y_true, y_pred, vocab):

        for i in range(len(y_true)):
            self.true_sentences.append(utils.generate_sentence(y_true[i].numpy().astype('int'), vocab))
            self.pred_sentences.append(utils.generate_sentence(np.argmax(y_pred[i].numpy(), axis=1).astype('int'), vocab))

    def result(self):
        return sacrebleu.corpus_bleu(self.pred_sentences, [self.true_sentences]).score

    def reset_states(self):
        self.true_sentences = []
        self.pred_sentences = []