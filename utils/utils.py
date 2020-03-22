import os
import typing

import numpy as np
import tensorflow as tf

UNKNOWN_TOKEN = '<unknown>'
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
    
    assert len(sorted_unique_words) >= vocab_size, "vocab_length is too big."
    
    # Build vocabulary
    vocab = {word:i for i, word in enumerate(sorted_unique_words[:vocab_size-1])}
    # vocab = {}
    # for i in range(vocab_size-1):
    #     vocab[sorted_unique_words[i]] = i#np.eye(vocab_size+1)[i]
    vocab[UNKNOWN_TOKEN] = vocab_size-1 #np.eye(vocab_size+1)[vocab_size]
        
    return vocab
    
def get_sentences(file_path: str) -> typing.List[typing.List[str]]:
    """Reads file and returns the sentences."""
    # Read file lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Split on words
    sentences = []
    for line in lines:
        sentences.append(line.strip().split())
        
    return sentences


def load_training_data(dir_path: str, 
                       vocab_en: typing.Dict[str, np.ndarray], 
                       vocab_fr: typing.Dict[str, np.ndarray], 
                       valid_ratio: float = 0.15,
                       max_seq_len: int = 20
                      ) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Returns train and valid datasets"""
    
    def sentence_to_vocab(sentences, vocab):
        # Build training data
        data = np.zeros((len(sentences), max_seq_len))
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if j >= max_seq_len:
                    break
                if sentences[i][j] in vocab:
                    data[i, j] = vocab[sentences[i][j]]
                else:
                    data[i, j] = vocab[UNKNOWN_TOKEN]
        return data
    
    # Get sentences
    sentences_en = get_sentences(os.path.join(dir_path, 'train.lang1'))
    sentences_fr = get_sentences(os.path.join(dir_path, 'train.lang2'))
    
    # Build training data
    train_X = sentence_to_vocab(sentences_en, vocab_en)
    train_y = sentence_to_vocab(sentences_fr, vocab_fr)
    
    # Split in train and valid
    cuttoff_idx = int(np.round(len(train_X)*valid_ratio))
    train_X, valid_X = train_X[:cuttoff_idx], train_X[cuttoff_idx:]
    train_y, valid_y = train_y[:cuttoff_idx], train_y[cuttoff_idx:]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_y))
    
    return train_dataset, valid_dataset