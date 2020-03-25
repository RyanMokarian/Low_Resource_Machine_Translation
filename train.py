import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable tensorflow debugging logs (Needs to be called before importing it)

import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models import baselines
from models.seq2seq_gru import Seq2SeqGRU
from utils import utils
from utils import logging
from utils import plots

SEED = 1

logger = logging.getLogger()

# Metrics
train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
valid_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss_function(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = cross_entropy(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

def train_epoch(model, data_loader, batch_size, optimizer):
    train_accuracy_metric.reset_states()
    for batch in tqdm(data_loader, desc='train epoch', leave=False):
        batch['gen_seq_len'] = batch['labels'].shape[1]
        inputs = batch['inputs']
        with tf.GradientTape() as tape:
            preds = model(batch, training=True)
            loss = loss_function(y_true=batch['labels'], y_pred=preds)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_accuracy_metric.update_state(y_true=batch['labels'], y_pred=preds)

def test_epoch(model, data_loader, batch_size, idx2word_fr):
    valid_accuracy_metric.reset_states()
    for batch in tqdm(data_loader, desc='train epoch', leave=False):
        batch['gen_seq_len'] = batch['labels'].shape[1]
        preds = model(batch)
        loss = loss_function(y_true=batch['labels'], y_pred=preds)

        valid_accuracy_metric.update_state(y_true=batch['labels'], y_pred=preds)

    label_sentence = utils.generate_sentence(batch['labels'][0].numpy().astype('int'), idx2word_fr)
    pred_sentence = utils.generate_sentence(np.argmax(preds[0].numpy().astype('int'), axis=1), idx2word_fr)
    logger.debug(f'Sample : \n  Label : {label_sentence}\n  Pred : {pred_sentence}')

def main(data_dir: str = '/project/cq-training-1/project2/teams/team12/data/',
         model: str = 'gru',
         epochs: int = 10,
         optimizer: str = 'adam',
         lr: float = 1e-4, 
         batch_size: int = 100,
         vocab_size: int = None, # If None all tokens of will be in vocab
         seq_len: int = 20, # If None the seq len is dynamic (might not work with all models)
         seed: bool = True
        ):
    # Call to remove tensorflow warning about casting float64 to float32
    tf.keras.backend.set_floatx('float32')

    # Set random seed
    if seed:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
        
    # Optimizer
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        raise Exception(f'Optimizer "{optimizer}" not recognized.')
        
    # Create vocabs
    logger.info('Creating vocab...')
    path_en = os.path.join(data_dir, 'train.lang1')
    path_fr = os.path.join(data_dir, 'train.lang2')
    word2idx_en, idx2word_en = utils.create_vocab(path_en, vocab_size)
    word2idx_fr, idx2word_fr = utils.create_vocab(path_fr, vocab_size)
    
    # Load datasets
    logger.info('Loading datasets...')
    train_dataset, valid_dataset = utils.load_training_data(path_en, path_fr, word2idx_en, word2idx_fr, seq_len, batch_size)

    # Create model
    if model == 'gru':
        model = baselines.GRU(len(word2idx_fr), batch_size)
    elif model == 'seq2seqgru':
        model = Seq2SeqGRU(len(word2idx_en), word2idx_fr, batch_size, embedding_dim=256, encoder_units=256, decoder_units=256, attention_units=256)
    else:
        raise Exception(f'Model "{model}" not recognized.')
    
    # Training loop
    logger.info('Training...')

    metrics = {'train_accuracy' : [], 'valid_accuracy' : []}
    best_valid_accuracy = 0
    for epoch in range(epochs):
        train_epoch(model, train_dataset, batch_size, optimizer)
        test_epoch(model, valid_dataset, batch_size, idx2word_fr)
        train_accuracy = np.sqrt(train_accuracy_metric.result().numpy())
        valid_accuracy = np.sqrt(valid_accuracy_metric.result().numpy())

        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            utils.save_model(model)
        
        # Logs
        logger.info(f'Epoch {epoch} - Train Accuracy : {train_accuracy:.4f}, Valid Accuracy : {valid_accuracy:.4f}')
        metrics['train_accuracy'].append(train_accuracy)
        metrics['valid_accuracy'].append(valid_accuracy)
            
    # Plot losses
    plots.plot_accuracy(metrics['train_accuracy'], metrics['valid_accuracy'])

if __name__ == "__main__":
    fire.Fire(main)