import random

from tqdm import tqdm
import numpy as np 
import tensorflow as tf
import pdb

from utils.utils import PADDING_TOKEN, UNKNOWN_TOKEN, sort
from utils import text_preprocessing
from utils import logging

logger = logging.getLogger()

def data_to_vocab(process_data,word2idx):
    data = []
    for sent in process_data:
        num_sent = []
        for token in sent:
            if token in word2idx:
                num_sent.append(word2idx[token])
            else:
                num_sent.append(word2idx[UNKNOWN_TOKEN])
        data.append(np.array(num_sent))
    return np.array(data)

# combine sentence and vocab to prevent textprocessing twice on large unaligned data 
def get_sentence_vocab(file_path: str, vocab_size: int):
    """Returns a dictionary that maps words to one hot embeddings"""
    
    # Get Sentences
    
    # Read file lines
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Split on words
    sentences = []
    for line in lines:
        line = text_preprocessing.process(line)
        sentences.append(line.split())
    
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
    return sentences, word2idx, idx2word


def load_data(data_path, vocab_size, valid_ratio, batch_size, seq_len=None):
    
    process_data, word2idx, idx2word = get_sentence_vocab(data_path,vocab_size=vocab_size)
    numerical = data_to_vocab(process_data, word2idx)
    
    # Split in train and valid
    cuttoff_idx = round(len(numerical)*(1-valid_ratio))
    train_data = numerical[:cuttoff_idx]
    valid_data = numerical[cuttoff_idx:]
    
    if not seq_len: 
        train_data = sort(train_data)
        valid_data = sort(valid_data)
        
    train_dataset = tf.data.Dataset.from_generator(lambda: [{'inputs':x[:-1], 'labels':x[1:]} for x in train_data], {'inputs':tf.int64, 'labels':tf.int64}, 
                                               output_shapes={'inputs':tf.TensorShape([None]), 'labels':tf.TensorShape([None])}) \
                               .shuffle(batch_size*3)\
                               .padded_batch(batch_size, drop_remainder=False, padded_shapes={'inputs':[seq_len], 'labels':[seq_len]})
    
    valid_dataset = tf.data.Dataset.from_generator(lambda: [{'inputs':x[:-1], 'labels':x[1:]} for x in valid_data], {'inputs':tf.int64, 'labels':tf.int64},
                                               output_shapes={'inputs':tf.TensorShape([None]), 'labels':tf.TensorShape([None])}) \
                               .padded_batch(batch_size, drop_remainder=False, padded_shapes={'inputs':[seq_len], 'labels':[seq_len]})
    
    return train_dataset, len(train_data), valid_dataset, len(valid_data), word2idx, idx2word



# Depreciate
def cont_batch_loader(raw_data,batch_sz,seq_len,is_train=False):
    
    if is_train:
        random.shuffle(raw_data)
        
    raw_data = [word for snt in raw_data for word in snt]
    batch_len = len(raw_data) // batch_sz
    
    # save data only if > =  seq_len
    if len(raw_data) - batch_len * batch_sz >= seq_len: 
        batch_len += 1
        
    data = np.zeros([batch_sz, batch_len], dtype = np.int32)
    # divide sequential data into required batches and pad
    for i in range(batch_sz):
        row = raw_data[i*batch_len:(i+1)*batch_len]
        diff =  batch_len - len(row) 
        row.extend([0]*diff) # will not do anything if batch_len = len(raw_data) // batch_sz
        data[i] = row 
        
    # divide each batch into chucks of seq_len
    for i in range((batch_len-1)//seq_len):
        x = data[:,i*seq_len:(i+1)*seq_len]
        y = data[:,i*seq_len+1:(i+1)*seq_len+1]
        yield(x,y)
        
    # adjust for last timestep when less than seq_len 
    if (batch_len-1)%seq_len !=0:
        x, y = data[:,(i+1)*seq_len:-1], data[:,(i+1)*seq_len+1:]
        if x.shape[1] >= (seq_len//3): # yield only if enough data
            yield(x,y)
