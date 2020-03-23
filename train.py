import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable tensorflow debugging logs (Needs to be called before importing it)

import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models import baselines
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
        inputs, labels = batch
        with tf.GradientTape() as tape:
            preds = model(inputs)
            loss = loss_function(y_true=labels, y_pred=preds)
            
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_accuracy_metric.update_state(y_true=labels, y_pred=preds)

def test_epoch(model, data_loader, batch_size):
    valid_accuracy_metric.reset_states()
    for batch in tqdm(data_loader, desc='train epoch', leave=False):
        inputs, labels = batch
        preds = model(inputs)
        loss = loss_function(y_true=labels, y_pred=preds)
            
        valid_accuracy_metric.update_state(y_true=labels, y_pred=preds)

def main(data_dir: str = '/project/cq-training-1/project2/data/',
         model: str = 'gru',
         epochs: int = 10,
         optimizer: str = 'adam',
         lr: float = 1e-4, 
         batch_size: int = 100,
         vocab_size: int = None, # If None all tokens of will be in vocab
         seq_len: int = 20,
         seed: bool = True
        ):
    # Call to remove tensorflow warning about casting float64 to float32
    tf.keras.backend.set_floatx('float64')

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
    vocab_en = utils.create_vocab(path_en, vocab_size)
    vocab_fr = utils.create_vocab(path_fr, vocab_size)
    
    # Load datasets
    logger.info('Loading datasets...')
    train_dataset, valid_dataset = utils.load_training_data(path_en, path_fr, vocab_en, vocab_fr, seq_len, batch_size)

    # Create model
    if model == 'gru':
        model = baselines.GRU(len(vocab_fr), batch_size)
    else:
        raise Exception(f'Model "{model}" not recognized.')
    
    # Training loop
    logger.info('Training...')

    metrics = {'train_accuracy' : [], 'valid_accuracy' : []}
    best_valid_accuracy = 0
    for epoch in range(epochs):
        train_epoch(model, train_dataset, batch_size, optimizer)
        test_epoch(model, valid_dataset, batch_size)
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