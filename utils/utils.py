import os
import typing
import pickle

import fasttext
import numpy as np
import tensorflow as tf

from utils import text_preprocessing
from utils import logging

logger = logging.getLogger()

PADDING_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'
SAVED_MODEL_DIR = 'saved_model'
SHARED_PATH = '/project/cq-training-1/project2/teams/team12/'

def create_folder(path: str):
    """ This function creates a folder if it does not already exists."""
    if not os.path.exists(path):
        os.mkdir(path)

def save_model(model: tf.keras.Model, name = None):
    """ This function saves the model to disk."""
    create_folder(SAVED_MODEL_DIR)
    if name: 
        model_path = os.path.join(SAVED_MODEL_DIR, name) 
    else: 
        model_path = os.path.join(SAVED_MODEL_DIR, model.get_name())
    create_folder(model_path)
    model.save_weights(os.path.join(model_path, "model"))

def save_metrics(metrics, name):
    """Save metrics to disk"""
    path = os.path.join(SAVED_MODEL_DIR, name)
    pickle.dump(metrics, open(os.path.join(path, 'metrics.pkl'), 'wb'))

def create_fasttext_embedding_matrix(file_path: str, vocab: typing.Dict[str, int]) -> typing.Dict[str, np.ndarray]:
    """Train a fasttext model and return the embeddings."""
    
    model_path = os.path.join(SHARED_PATH, 'embedding_models', 'fasttext_model.bin')
    
    if os.path.exists(model_path):
        logger.info('Loading fasttext embeddings...')
        model = fasttext.load_model(model_path)
    else:
        logger.info('Training fasttext embeddings...')
        model = fasttext.train_unsupervised(file_path, model='skipgram')
        model.save_model(model_path)

    embedding_matrix = np.zeros((len(vocab), model.get_dimension()))
    for word in vocab.keys():
        idx = vocab[word]
        if word in model.words:
            embedding_matrix[idx] = model[word]
        else:
            pass # If word embedding is unknown, vector of zeros

    return embedding_matrix

def create_vocab(file_path: str, vocab_size: int) -> typing.Dict[str, np.ndarray]:
    """Returns a dictionary that maps words to one hot embeddings"""
    # Get sentences
    sentences = get_sentences(file_path)
    
    # Get words
    words = []
    for sentence in sentences:
        words.extend(sentence)
    
    # Get unique words
    unique_words, word_counts = np.unique(words, return_counts=True)
    sorted_unique_words = unique_words[np.argsort(word_counts)[::-1]]
    if vocab_size is None: 
        vocab_size = len(sorted_unique_words)
    if vocab_size > len(sorted_unique_words):
        vocab_size = len(sorted_unique_words)
        logger.info(f"vocab_size is too big. Using vocab_size = {vocab_size} ")
    
    # Build vocabulary
    word2idx = {word:i+1 for i, word in enumerate(sorted_unique_words[:vocab_size])}
    word2idx[PADDING_TOKEN] = 0 
    word2idx[UNKNOWN_TOKEN] = vocab_size + 1 
    idx2word = {i+1:word for i, word in enumerate(sorted_unique_words[:vocab_size])}
    idx2word[0] = PADDING_TOKEN
    idx2word[vocab_size+1] = UNKNOWN_TOKEN

    return word2idx, idx2word
    
def get_sentences(file_path: str) -> typing.List[typing.List[str]]:
    """Reads file and returns the sentences."""
    # Read file lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Split on words
    sentences = []
    for line in lines:
        line = text_preprocessing.process(line)
        sentences.append(line.split())
        
    return sentences

def sort(x,y=None):
    """ Sort data according to len when using dynamic seq_len for efficient batching."""
    idx = np.argsort([len(ip) for ip in x])[::-1]
    if y == None:
        return x[idx]
    return x[idx], y[idx]


def load_training_data(en_path: str,
                       fr_path: str, 
                       vocab_en: typing.Dict[str, np.ndarray], 
                       vocab_fr: typing.Dict[str, np.ndarray],
                       seq_len: int,
                       batch_size: int,
                       valid_ratio: float = 0.15,
                      ) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Returns train and valid datasets"""
    
    def sentence_to_vocab(sentences, vocab):
        data = []
        for i in range(len(sentences)):
            sentence = []
            for j in range(len(sentences[i])):
                if seq_len is not None and j >= seq_len:
                    break 
                if sentences[i][j] in vocab:
                    sentence.append(vocab[sentences[i][j]])
                else:
                    sentence.append(vocab[UNKNOWN_TOKEN])
            data.append(np.array(sentence))
        return np.array(data)
    # Get sentences
    sentences_en = get_sentences(en_path)
    sentences_fr = get_sentences(fr_path)
    
    # Build training data
    train_X = sentence_to_vocab(sentences_en, vocab_en)
    train_y = sentence_to_vocab(sentences_fr, vocab_fr)
    
    # Split in train and valid
    cuttoff_idx = int(np.round(len(train_X)*(1-valid_ratio)))
    train_X, valid_X = train_X[:cuttoff_idx], train_X[cuttoff_idx:]
    train_y, valid_y = train_y[:cuttoff_idx], train_y[cuttoff_idx:]
    
    if not seq_len: 
        train_X, train_y = sort(train_X, train_y)
        valid_X, valid_y = sort(valid_X, valid_y)

    train_dataset = tf.data.Dataset.from_generator(lambda: [{'inputs':x, 'labels':y} for x, y in zip(train_X, train_y)], {'inputs':tf.int64, 'labels':tf.int64}, 
                                                   output_shapes={'inputs':tf.TensorShape([None]), 'labels':tf.TensorShape([None])}) \
                                   .shuffle(batch_size*3)\
                                   .padded_batch(batch_size, drop_remainder=False, padded_shapes={'inputs':[seq_len], 'labels':[seq_len]})
    valid_dataset = tf.data.Dataset.from_generator(lambda:  [{'inputs':x, 'labels':y} for x, y in zip(valid_X, valid_y)], {'inputs':tf.int64, 'labels':tf.int64},
                                                   output_shapes={'inputs':tf.TensorShape([None]), 'labels':tf.TensorShape([None])}) \
                                   .padded_batch(batch_size, drop_remainder=False, padded_shapes={'inputs':[seq_len], 'labels':[seq_len]})

    return train_dataset, valid_dataset, len(train_y), len(valid_y)

def generate_sentence(indices: typing.List[int], vocab: typing.Dict[int, str]):
    """Generate a sentence from a list of indices."""
    sentence = ''
    for idx in indices:
        if int(idx) not in vocab:
            print(f'idx {idx} not in vocab')
            continue
        elif vocab[idx] == PADDING_TOKEN \
            or vocab[idx] == text_preprocessing.BOS:
            continue
        elif vocab[idx] == text_preprocessing.EOS:
            break

        sentence += vocab[int(idx)]
        sentence += ' '
    
    sentence = text_preprocessing.recapitalize(sentence)

    return sentence

# Code from : https://www.tensorflow.org/tutorials/text/transformer
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate scheduler for the transformer."""
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)