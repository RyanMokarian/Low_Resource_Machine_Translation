
import sacrebleu
import numpy as np
import tensorflow as tf
import tf.keras.backend as K  # Alias to Keras' backend namespace.

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


# copied over from   https://gist.github.com/Gregorgeous/dbad1ec22efc250c76354d949a13cec3
class PerplexityMetric(tf.keras.metrics.Metric):
    """
    USAGE NOTICE: this metric accepts only logits for now (i.e. expect the same behaviour as from tf.keras.losses.SparseCategoricalCrossentropy with the a provided argument "from_logits=True", 
		here the same loss is used with "from_logits=True" enforced so you need to provide it in such a format)
    METRIC DESCRIPTION:
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf.
    DISCLAIMER: Original function created by Kirill Mavreshko in https://github.com/kpot/keras-transformer/blob/b9d4e76c535c0c62cadc73e37416e4dc18b635ca/example/run_gpt.py#L106. 
    My "contribution": I converted Kirill method's logic (and added a padding masking to to it) into this new Tensorflow 2.0 way of doing things via a stateful "Metric" object. This required making the metric a fully-fledged object by subclassing the Metric class. 
    """
    def __init__(self, name='perplexity', **kwargs):
      super(PerplexityMetric, self).__init__(name=name, **kwargs)
      self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
      self.perplexity = self.add_weight(name='tp', initializer='zeros')

		# Consider uncommenting the decorator for a performance boost (?)  		
    # @tf.function
    def _calculate_perplexity(self, real, pred):
			# The next 4 lines zero-out the padding from loss calculations, 
			# this follows the logic from: https://www.tensorflow.org/beta/tutorials/text/transformer#loss_and_metrics 			
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = self.cross_entropy(real, pred)
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
			# Calculating the perplexity steps: 			
      step1 = K.mean(loss_, axis=-1)
      step2 = K.exp(step1)
      perplexity = K.mean(step2)

      return perplexity 


    def update_state(self, y_true, y_pred, sample_weight=None):
      # TODO:FIXME: handle sample_weight ! 
      if sample_weight is not None:
          print("WARNING! Provided 'sample_weight' argument to the perplexity metric. Currently this is not handled and won't do anything differently..")
      perplexity = self._calculate_perplexity(y_true, y_pred)
			# Remember self.perplexity is a tensor (tf.Variable), so using simply "self.perplexity = perplexity" will result in error because of mixing EagerTensor and Graph operations 
      self.perplexity.assign_add(perplexity)
        
    def result(self):
      return self.perplexity

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.perplexity.assign(0.)