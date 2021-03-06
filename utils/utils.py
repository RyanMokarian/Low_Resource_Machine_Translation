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
MAX_SEQ_LEN = 134 # Maximum sequence length of the aligned data

aligned_data = None
back_translated_data = None


def create_folder(path: str):
    """ This function creates a folder if it does not already exists."""
    if not os.path.exists(path):
        os.mkdir(path)


def save_model(model: tf.keras.Model, name: str = None):
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


def create_fasttext_embedding_matrix(file_path: str, vocab: typing.Dict[str, int],
                                     embedding_dim: int) -> typing.Dict[str, np.ndarray]:
    """Train a fasttext model and return the embeddings."""

    model_path = os.path.join(SHARED_PATH, 'embedding_models', f'fasttext_model_dim_{embedding_dim}.bin')

    if os.path.exists(model_path):
        logger.info('Loading fasttext embeddings...')
        model = fasttext.load_model(model_path)
    else:
        logger.info('Training fasttext embeddings...')
        model = fasttext.train_unsupervised(file_path, model='skipgram', dim=embedding_dim)
        model.save_model(model_path)

    embedding_matrix = np.zeros((len(vocab), model.get_dimension()))
    for word in vocab.keys():
        idx = vocab[word]
        if word in model.words:
            embedding_matrix[idx] = model[word]
        else:
            pass  # If word embedding is unknown, vector of zeros

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
    word2idx = {word: i + 1 for i, word in enumerate(sorted_unique_words[:vocab_size])}
    word2idx[PADDING_TOKEN] = 0
    word2idx[UNKNOWN_TOKEN] = vocab_size + 1
    idx2word = {i + 1: word for i, word in enumerate(sorted_unique_words[:vocab_size])}
    idx2word[0] = PADDING_TOKEN
    idx2word[vocab_size + 1] = UNKNOWN_TOKEN

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
        sentence = line.split()
        if len(sentence) > MAX_SEQ_LEN:
            sentence = sentence[:MAX_SEQ_LEN]
        sentences.append(sentence)

    return sentences


def sort(x, y=None):
    """ Sort data according to len when using dynamic seq_len for efficient batching."""
    idx = np.argsort([len(ip) for ip in x])[::-1]
    if y == None:
        return x[idx]
    return x[idx], y[idx]


def load_data(path: str, vocab: typing.Dict[str, np.ndarray], seq_len: int = None) -> np.ndarray:
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

    sentences = get_sentences(path)
    data = sentence_to_vocab(sentences, vocab)

    return data


def load_training_data(en_path: str,
                       fr_path: str,
                       vocab_en: typing.Dict[str, np.ndarray],
                       vocab_fr: typing.Dict[str, np.ndarray],
                       seq_len: int,
                       batch_size: int,
                       valid_ratio: float = 0.15,
                       fr_unaligned_path: str = None,
                       en_back_translated_path: str = None,
                       back_translation_ratio: float = 1.0) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Returns train and valid datasets"""

    # Global variables that hold the data to avoid reloading it multiple times when doing back-translation
    # (We load a new training set each epoch when doing back-translation)
    global aligned_data
    global back_translated_data

    # Build training data
    if aligned_data is None:
        train_X = load_data(en_path, vocab_en, seq_len)
        train_y = load_data(fr_path, vocab_fr, seq_len)
        aligned_data = (train_X, train_y)
    else:
        train_X, train_y = aligned_data

    # Split in train and valid
    cuttoff_idx = int(np.round(len(train_X) * (1 - valid_ratio)))
    train_X, valid_X = train_X[:cuttoff_idx], train_X[cuttoff_idx:]
    train_y, valid_y = train_y[:cuttoff_idx], train_y[cuttoff_idx:]

    logger.debug(f'shape train_X : {train_X.shape}')
    logger.debug(f'shape train_y : {train_y.shape}')

    # Load back-translated data if available
    if fr_unaligned_path is not None and en_back_translated_path is not None:

        if back_translated_data is None:
            back_translated_X = load_data(en_back_translated_path, vocab_en, seq_len)
            unaligned_y = load_data(fr_unaligned_path, vocab_fr, seq_len)
            back_translated_data = (back_translated_X, unaligned_y)
        else:
            back_translated_X, unaligned_y = back_translated_data

        # Sample data according to back translation ratio
        nb_examples = int(len(train_X) * back_translation_ratio)
        sample = np.random.randint(0, len(back_translated_X), nb_examples)
        back_translated_X = back_translated_X[sample]
        unaligned_y = unaligned_y[sample]

        train_X = np.concatenate((train_X, back_translated_X), axis=0)
        train_y = np.concatenate((train_y, unaligned_y), axis=0)

    logger.debug(f'shape train_X : {train_X.shape}')
    logger.debug(f'shape train_y : {train_y.shape}')

    if not seq_len:
        train_X, train_y = sort(train_X, train_y)
        valid_X, valid_y = sort(valid_X, valid_y)

    train_dataset = tf.data.Dataset.from_generator(lambda: [{'inputs':x,'labels':y} for x, y in zip(train_X, train_y)],
                                                   output_types={'inputs':tf.int64, 'labels':tf.int64},
                                                   output_shapes={'inputs':tf.TensorShape([None]), 
                                                                  'labels':tf.TensorShape([None])})\
                                   .shuffle(batch_size*3)\
                                   .padded_batch(batch_size, 
                                                 drop_remainder=False, 
                                                 padded_shapes={'inputs':[seq_len], 'labels':[seq_len]})

    valid_dataset = tf.data.Dataset.from_generator(lambda: [{'inputs':x,'labels':y} for x, y in zip(valid_X, valid_y)],
                                           output_types={'inputs':tf.int64, 'labels':tf.int64},
                                           output_shapes={'inputs':tf.TensorShape([None]), 
                                                          'labels':tf.TensorShape([None])}) \
                                   .padded_batch(batch_size, 
                                                 drop_remainder=False, 
                                                 padded_shapes={'inputs':[seq_len], 'labels':[seq_len]})

    return train_dataset, valid_dataset, len(train_y), len(valid_y)


def generate_sentence(indices: typing.List[int], vocab: typing.Dict[int, str], ignore_unknown: bool = True) -> str:
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


def generate_sentence_from_probabilities(probs: typing.List[np.ndarray],
                                         vocab: typing.Dict[int, str],
                                         ignore_unknown: bool = True) -> str:
    """Generate a sentence from a list of probability vector."""
    indices = np.argmax(probs, axis=1).astype('int')
    sentence = ''
    for i, idx in enumerate(indices):
        if int(idx) not in vocab:
            print(f'idx {idx} not in vocab')
            continue
        if vocab[idx] == UNKNOWN_TOKEN and ignore_unknown:
            idx = int(np.argsort(probs[i])[-2])  # Take the second biggest prob
        if vocab[idx] == PADDING_TOKEN \
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
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
