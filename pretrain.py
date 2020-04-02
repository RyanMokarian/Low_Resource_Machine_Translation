import math
import os
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import fire
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm

import utils.pretrain_utils as pre_utils
from models import baselines
from utils import utils
from utils import logging
from utils import plots
from utils import metrics
from train import cross_entropy, loss_function, train_accuracy_metric, valid_accuracy_metric

SEED = 1

logging.initializeLogger()
logger = logging.getLogger()

train_loss_metric = metrics.Perplexity()
valid_loss_metric = metrics.Perplexity()


def train_epoch(model, data_loader, optimizer, batch_nb):
    train_accuracy_metric.reset_states()
    train_loss_metric.reset_states()
    for batch in tqdm(data_loader, total=batch_nb, desc='train epoch', leave=False):
        x, y_true = batch['inputs'], batch['labels']
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        with tf.GradientTape() as tape:
            preds = model(x, training=True)
            loss = loss_function(y_true, preds, mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_accuracy_metric.update_state(y_true, preds, sample_weight=mask)
        train_loss_metric.update_state(y_true, preds)


def test_epoch(model, data_loader, epoch, batch_nb):
    valid_accuracy_metric.reset_states()
    valid_loss_metric.reset_states()
    for batch in tqdm(data_loader, total=batch_nb, desc='valid epoch', leave=False):
        x, y_true = batch['inputs'], batch['labels']
        mask = tf.math.logical_not(tf.math.equal(y_true, 0))
        preds = model(x, training=False)
        valid_accuracy_metric.update_state(y_true, preds, sample_weight=mask)
        valid_loss_metric.update_state(y_true, preds)
    if epoch >= 5:
        idx = np.random.choice(range(10))
        store, _ = model.generate(x[idx, :7][None])
        logger.debug(
            f"Starting from '{utils.generate_sentence(x[idx,:7].numpy(), idx2word)}' \n '{utils.generate_sentence(store, idx2word)}' "
        )


def main(
    data: str = 'train_fr',  # extra/train .en/.fr  
    model: str = 'lstm',
    layers: int = 1,
    epochs: int = 10,
    optimizer: str = 'adam',
    lr: float = 3e-4,
    batch_size: int = 64,
    vocab_size: int = 20000,
    seq_len: int = None,
    seed: bool = True):

    # Call to remove tensorflow warning about casting float64 to float32
    tf.keras.backend.set_floatx('float32')

    path = '/project/cq-training-1/project2/teams/team12/data/'
    # data file
    if data == 'extra_fr': f = 'unaligned-tok.fr'
    elif data == 'extra_en': f = 'unaligned-tok.en'
    elif data == 'train_en': f = 'train.lang1'
    elif data == 'train_fr': f = 'train.lang2'
    else: raise Exception(f'Data file: "{data}" not recognized.')
    path = path + f

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

    # Load datasets
    logger.info('Loading datasets...')
    train_dataset, len_train, valid_dataset, len_valid, word2idx, idx2word = pre_utils.load_data(path, \
                                                                 vocab_size, 0.1, batch_size, seq_len)

    logger.info(f'Number of training examples : {len_train}, number of valid examples : {len_valid}')

    # Create model
    if model == 'lstm':
        model = baselines.RecurrentNet(vocab_size, embed_dims=128, rnn_type='lstm', num_layers=layers)
    elif model == 'gru':
        model = baselines.RecurrentNet(vocab_size, embed_dims=128, rnn_type='gru', num_layers=layers)
    else:
        raise Exception(f'Model "{model}" not recognized.')

    # to differentiate models
    model_name = model.__class__.__name__+'_'+str(model.vocab_size)+'_'+str(model.embed_dims) + \
                                            '_'+str(model.rnn_type) +'_'+str(model.num_layers)

    # Training loop
    logger.info('Training...')
    metrics = {'train_accuracy': [], 'valid_accuracy': [], 'train_loss': [], 'valid_loss': []}
    best_valid_loss = float('inf')
    for epoch in range(epochs):
        train_epoch(model, train_dataset, optimizer, math.ceil(len_train / batch_size))
        test_epoch(model, valid_dataset, epoch, math.ceil(len_valid / batch_size))
        train_accuracy = train_accuracy_metric.result().numpy()
        valid_accuracy = valid_accuracy_metric.result().numpy()
        train_loss = train_loss_metric.result().numpy()
        valid_loss = valid_loss_metric.result().numpy()
        logger.info(
            f'Epoch {epoch}\n    Train Loss : {train_loss:.4f} - Valid Loss : {valid_loss:.4f}\n    Train Accuracy : {train_accuracy:.4f} - Valid Accuracy : {valid_accuracy:.4f}'
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            utils.save_model(model, model_name)

        metrics['train_accuracy'].append(train_accuracy)
        metrics['valid_accuracy'].append(valid_accuracy)
        metrics['train_loss'].append(train_loss)
        metrics['valid_loss'].append(valid_loss)

    # save metrics
    utils.save_metrics(metrics, model_name)


if __name__ == '__main__':
    fire.Fire(main)
