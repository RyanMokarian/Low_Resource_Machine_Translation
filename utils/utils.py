import os
import typing

import numpy as np
import tensorflow as tf

from utils import text_preprocessing

PADDING_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'
SAVED_MODEL_DIR = 'saved_model'

def create_folder(path: str):
    """ This function creates a folder if it does not already exists."""
    if not os.path.exists(path):
        os.mkdir(path)

def save_model(model: tf.keras.Model):
    """ This function saves the model to disk."""
    create_folder(SAVED_MODEL_DIR)
    model_path = os.path.join(SAVED_MODEL_DIR, model.__class__.__name__)
    create_folder(model_path)
    model.save_weights(os.path.join(model_path, "model"))

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
    assert vocab_size <= len(sorted_unique_words), "vocab_size is too big."
    
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
    
    train_dataset = tf.data.Dataset.from_generator(lambda: [{'inputs':x, 'labels':y} for x, y in zip(train_X, train_y)], {'inputs':tf.int64, 'labels':tf.int64}, 
                                                   output_shapes={'inputs':tf.TensorShape([None]), 'labels':tf.TensorShape([None])}) \
                                   .padded_batch(batch_size, drop_remainder=True, padded_shapes={'inputs':[seq_len], 'labels':[seq_len]})
    valid_dataset = tf.data.Dataset.from_generator(lambda:  [{'inputs':x, 'labels':y} for x, y in zip(valid_X, valid_y)], {'inputs':tf.int64, 'labels':tf.int64},
                                                   output_shapes={'inputs':tf.TensorShape([None]), 'labels':tf.TensorShape([None])}) \
                                   .padded_batch(batch_size, drop_remainder=True, padded_shapes={'inputs':[seq_len], 'labels':[seq_len]})

    return train_dataset, valid_dataset

def generate_sentence(indices: typing.List[int], vocab: typing.Dict[int, str]):
    """Generate a sentence from a list of indices."""
    sentence = ''
    for idx in indices:
        if int(idx) not in vocab:
            print(f'idx {idx} not in vocab')
            continue
        elif vocab[idx] == PADDING_TOKEN \
            or vocab[idx] == text_preprocessing.BOS \
            or vocab[idx] == text_preprocessing.EOS:
            continue

        sentence += vocab[int(idx)]
        sentence += ' '
    
    sentence = text_preprocessing.recapitalize(sentence)

    return sentence