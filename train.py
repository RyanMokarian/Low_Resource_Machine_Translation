import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Disable tensorflow debugging logs (Needs to be called before importing it)

import fire
import numpy as np
import tensorflow as tf

from models import baselines
from utils import utils
from utils import logging

SEED = 1

logger = logging.getLogger()

def main(data_dir: str = '/project/cq-training-1/project2/data/',
         model: str = 'dummy',
         vocab_size: str = 1000,
         optimizer: str = 'adam',
         lr: float = 1e-4, 
         seed: bool = True
        ):
    
    # Set random seed
    if seed:
        tf.random.set_seed(SEED)
        np.random.seed(SEED)
    
    # Create model
    if model == 'dummy':
        model = baselines.DummyModel(vocab_size)
    else:
        raise Exception(f'Model "{model}" not recognized.')
        
    # Loss and optimizer
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    if optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr)
    elif optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        raise Exception(f'Optimizer "{optimizer}" not recognized.')
        
    # Create vocabs
    vocab_en = utils.create_vocab(os.path.join(data_dir, 'train.lang1'), vocab_size)
    vocab_fr = utils.create_vocab(os.path.join(data_dir, 'train.lang2'), vocab_size)
    
    # Load datasets
    train_dataset, valid_dataset = utils.load_training_data(data_dir, vocab_en, vocab_fr)
    
    # Training loop
    logger.info('Training...')

    

if __name__ == "__main__":
    fire.Fire(main)