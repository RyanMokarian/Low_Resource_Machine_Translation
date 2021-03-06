import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable tensorflow debugging logs (Needs to be called before importing it)

import fire
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from models import baselines
from models.seq2seq_gru import Seq2SeqGRU
from models.transformer import Transformer
from utils import utils
from utils import logging
from utils import plots
from utils import metrics

SEED = 1

logging.initializeLogger()
logger = logging.getLogger()

# Metrics
train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
valid_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()
train_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy()
valid_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy()
train_bleu_metric = metrics.BleuScore()
valid_bleu_metric = metrics.BleuScore()
cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


def loss_function(y_true, y_pred, mask):
    loss_ = cross_entropy(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    loss_ = tf.reduce_sum(loss_) / tf.reduce_sum(mask)  # prevent taking average over padding positions as well
    return loss_


def train_epoch(model, data_loader, optimizer, batch_nb, idx2word_fr):
    train_accuracy_metric.reset_states()
    train_loss_metric.reset_states()
    train_bleu_metric.reset_states()
    for batch in tqdm(data_loader, total=batch_nb, desc='train epoch', leave=False):
        labels = batch['labels']
        batch['gen_seq_len'] = labels.shape[1]
        with tf.GradientTape() as tape:
            preds = model(batch, training=True)
            labels = labels[:, 1:]  # Ignore BOS token (already not in preds)
            mask = tf.math.logical_not(tf.math.equal(labels, 0))
            loss = loss_function(y_true=labels, y_pred=preds, mask=mask)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_accuracy_metric.update_state(y_true=labels, y_pred=preds, sample_weight=mask)
        train_loss_metric.update_state(y_true=labels, y_pred=preds, sample_weight=mask)
        train_bleu_metric.update_state(y_true=labels, y_pred=preds, vocab=idx2word_fr)


def test_epoch(model, data_loader, batch_nb, idx2word_fr, idx2word_en):
    valid_accuracy_metric.reset_states()
    valid_loss_metric.reset_states()
    valid_bleu_metric.reset_states()
    for batch in tqdm(data_loader, total=batch_nb, desc='valid epoch', leave=False):
        labels = batch['labels']
        batch['gen_seq_len'] = labels.shape[1]

        preds = model(batch)
        labels = labels[:, 1:]  # Ignore BOS token (already not in preds)
        mask = tf.math.logical_not(tf.math.equal(labels, 0))

        loss = loss_function(y_true=labels, y_pred=preds, mask=mask)

        valid_accuracy_metric.update_state(y_true=labels, y_pred=preds, sample_weight=mask)
        valid_loss_metric.update_state(y_true=labels, y_pred=preds, sample_weight=mask)
        valid_bleu_metric.update_state(y_true=labels, y_pred=preds, vocab=idx2word_fr)

    idx = 0
    label_sentence = utils.generate_sentence(labels[idx].numpy(), idx2word_fr)
    pred_sentence = utils.generate_sentence_from_probabilities(preds[idx].numpy(), idx2word_fr)
    source_sentence = utils.generate_sentence(batch['inputs'][idx].numpy().astype('int'), idx2word_en)
    logger.debug(f'Sample : \n    Source : {source_sentence}\n    Pred : {pred_sentence}\n    Label : {label_sentence}')


def main(
    data_dir: str = '/project/cq-training-1/project2/teams/team12/data/',
    model_name: str = 'seq2seqgru',
    epochs: int = 20,
    optimizer: str = 'adam',
    lr: float = 1e-3,
    batch_size: int = 32,
    vocab_size: int = None,  # If None all tokens of will be in vocab
    seq_len: int = None,  # If None the seq len is dynamic (might not work with all models)
    seed: bool = True,
    model_config: dict = None,
    embedding: str = None,
    embedding_dim: int = 128,
    back_translation_model: str = 'saved_model/<model_folder_name>',
    back_translation: bool = False,
    back_translation_ratio: float = 1.0,
    fr_to_en: bool = False):

    # Call to remove tensorflow warning about casting float64 to float32
    tf.keras.backend.set_floatx('float32')

    # Set random seed
    if seed:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)

    # Data paths
    path_en = os.path.join(data_dir, 'train.lang1')
    path_fr = os.path.join(data_dir, 'train.lang2')
    path_unaligned_en = os.path.join(data_dir, 'unaligned-tok.en')
    path_unaligned_fr = os.path.join(data_dir, 'unaligned-tok.fr')
    if fr_to_en:  # Switch paths
        tmp = path_en
        path_en = path_fr
        path_fr = tmp

    # Create vocabs
    logger.info('Creating vocab...')
    word2idx_en, idx2word_en = utils.create_vocab(path_en, vocab_size)
    word2idx_fr, idx2word_fr = utils.create_vocab(path_fr, vocab_size)
    logger.info(f'Size of english vocab : {len(word2idx_en)}, size of french vocab : {len(word2idx_fr)}')

    # Back translation
    prediction_file = None
    if back_translation:
        prediction_file = os.path.join(utils.SHARED_PATH, 'translated_unaligned.en')
        if os.path.exists(prediction_file):
            logger.info(f'Using translation from {prediction_file} for back-translation.')
        else:
            logger.info(f'Translating {path_unaligned_fr} for back-translation...')
            # Load data
            data = utils.load_data(path_unaligned_fr, word2idx_fr)
            dataset = tf.data.Dataset.from_generator(lambda: [ex for ex in data],
                                                     tf.int64,
                                                     output_shapes=tf.TensorShape([None])).padded_batch(
                                                         128, padded_shapes=[None])
            # Load model
            model_config = {'num_layers': 2, 'd_model': 128, 'dff': 512, 'num_heads': 8}
            model = Transformer(model_config, len(word2idx_fr), word2idx_en)
            model.load_weights(os.path.join(back_translation_model, "model"))

            # Write prediction to file
            with open(prediction_file, 'w') as f:
                print('opening file and writing predictions...')
                for batch in tqdm(dataset, desc='Translating...', total=len(data) // 128 + 1):
                    preds = model({'inputs': batch, 'labels': tf.zeros_like(batch)})
                    for pred in preds:
                        sentence = utils.generate_sentence(np.argmax(pred.numpy(), axis=1).astype('int'), idx2word_en)
                        f.writelines([sentence, '\n'])

    # Load datasets
    logger.info('Loading datasets...')
    train_dataset, valid_dataset, nb_train_ex, nb_valid_ex = utils.load_training_data(
        path_en,
        path_fr,
        word2idx_en,
        word2idx_fr,
        seq_len,
        batch_size,
        en_back_translated_path=prediction_file,
        fr_unaligned_path=path_unaligned_fr,
        back_translation_ratio=back_translation_ratio)
    logger.info(f'Number of training examples : {nb_train_ex}, number of valid examples : {nb_valid_ex}')

    # Load embeddings
    embedding_matrix = None
    if embedding:
        logger.info(f'Loading embedding {embedding} ...')
        if embedding == 'fasttext':
            embedding_matrix = utils.create_fasttext_embedding_matrix(path_unaligned_en, word2idx_en, embedding_dim)
        elif embedding == 'word2vec':
            raise Exception(f'Embedding "{embedding}" not implemented yet')
        elif embedding == 'glove':
            raise Exception(f'Embedding "{embedding}" not implemented yet')
        else:
            raise Exception(f'Embedding "{embedding}" not recognized.')

    # Create model
    if model_name == 'gru':
        model = baselines.GRU(len(word2idx_fr), batch_size)
    elif model_name == 'seq2seqgru':
        if model_config is None:
            model_config = {'embedding_dim': 256, 'encoder_units': 512, 'decoder_units': 512, 'n_layers': 1}
        model = Seq2SeqGRU(len(word2idx_en), word2idx_fr, batch_size, model_config, embedding_matrix=embedding_matrix)
    elif model_name == 'transformer':
        if model_config is None:
            model_config = {'num_layers': 2, 'd_model': 128, 'dff': 512, 'num_heads': 8}
        model = Transformer(model_config, len(word2idx_en), word2idx_fr, embedding_matrix=embedding_matrix)
    else:
        raise Exception(f'Model "{model}" not recognized.')

    # Optimizer
    if optimizer == 'adam':
        if model_name == 'transformer':  # Use adam according to transformer paper
            optimizer = tf.keras.optimizers.Adam(utils.CustomSchedule(model_config['d_model']),
                                                 beta_1=0.9,
                                                 beta_2=0.98,
                                                 epsilon=1e-9)
            logger.info('Using custom scheduler for learning rate, --lr argument ignored.')
        else:
            optimizer = tf.keras.optimizers.Adam(lr)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        raise Exception(f'Optimizer "{optimizer}" not recognized.')

    # Training loop
    logger.info(f'Training with model {model.get_name()} ...')

    metrics = {
        'train_accuracy': [],
        'valid_accuracy': [],
        'train_loss': [],
        'valid_loss': [],
        'train_bleu': [],
        'valid_bleu': []
    }
    model_path = model.get_name() + f'_fr_to_en_{fr_to_en}_embedding_{embedding}_embedding_dim_{embedding_dim}'\
                                    f'_back_translation_{back_translation}_ratio_{back_translation_ratio}'
    best_valid_bleu = 0
    for epoch in range(epochs):
        train_epoch(model, train_dataset, optimizer, np.ceil(nb_train_ex / batch_size), idx2word_fr)
        test_epoch(model, valid_dataset, np.ceil(nb_valid_ex / batch_size), idx2word_fr, idx2word_en)
        train_accuracy = train_accuracy_metric.result().numpy()
        valid_accuracy = valid_accuracy_metric.result().numpy()
        train_loss = train_loss_metric.result().numpy()
        valid_loss = valid_loss_metric.result().numpy()
        train_bleu = train_bleu_metric.result()
        valid_bleu = valid_bleu_metric.result()

        if valid_bleu > best_valid_bleu:
            best_valid_bleu = valid_bleu
            utils.save_model(model, model_path)

        # Logs
        logger.info(f'Epoch {epoch}\n'\
                    f'    Train BLEU : {train_bleu:.4f} - Valid BLEU : {valid_bleu:.4f}\n'\
                    f'    Train Accuracy : {train_accuracy:.4f} - Valid Accuracy : {valid_accuracy:.4f}\n'\
                    f'    Train Loss : {train_loss:.4f} - Valid Loss : {valid_loss:.4f}')

        metrics['train_accuracy'].append(train_accuracy)
        metrics['valid_accuracy'].append(valid_accuracy)
        metrics['train_loss'].append(train_loss)
        metrics['valid_loss'].append(valid_loss)
        metrics['train_bleu'].append(train_bleu)
        metrics['valid_bleu'].append(valid_bleu)

        # If using back translation, sample new generated examples for next epoch
        if back_translation:
            train_dataset, _, _, _ = utils.load_training_data(path_en,
                                                              path_fr,
                                                              word2idx_en,
                                                              word2idx_fr,
                                                              seq_len,
                                                              batch_size,
                                                              en_back_translated_path=prediction_file,
                                                              fr_unaligned_path=path_unaligned_fr,
                                                              back_translation_ratio=back_translation_ratio)

        # If training with embeddings, unfreeze embedding layer at 50th epoch
        if epoch == 48 and embedding and model_name == 'transformer':
            model.unfreeze_embedding_layer()

    # save metrics
    utils.save_metrics(metrics, model_path)

    # Plot accuracy
    plots.plot_accuracy(metrics['train_accuracy'], metrics['valid_accuracy'])


if __name__ == "__main__":
    fire.Fire(main)