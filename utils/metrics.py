
import sacrebleu
import numpy as np
import tensorflow as tf

from utils import utils

class BleuScore():
    """This class computes the BLEU Metric."""
    def __init__(self):
        self.total_score = 0
        self.total_num_examples = 0

    def update_state(self, y_true, y_pred, vocab):
        for i in range(len(y_true)):
            label_sentence = utils.generate_sentence(y_true[i].numpy().astype('int'), vocab)
            pred_sentence = utils.generate_sentence(np.argmax(y_pred[i].numpy(), axis=1).astype('int'), vocab)
            self.total_score += sacrebleu.sentence_bleu(pred_sentence, label_sentence).score
            self.total_num_examples += 1

    def result(self):
        return self.total_score / self.total_num_examples

    def reset_states(self):
        self.total_score = 0
        self.total_num_examples = 0