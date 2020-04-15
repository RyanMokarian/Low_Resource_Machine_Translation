import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import subprocess
import tempfile

from tqdm import tqdm
import numpy as np
import tensorflow as tf

from models.transformer import Transformer
from utils import logging
from utils import utils

logging.initializeLogger()
logger = logging.getLogger()


def generate_predictions(input_file_path: str, pred_file_path: str):
    """Generates predictions for the machine translation task (EN->FR).

    You are allowed to modify this function as needed, but one again, you cannot
    modify any other part of this file. We will be importing only this function
    in our final evaluation script. Since you will most definitely need to import
    modules for your code, you must import these inside the function itself.

    Args:
        input_file_path: the file path that contains the input data.
        pred_file_path: the file path where to store the predictions.

    Returns: None

    """

    ##### MODIFY BELOW #####
    data_dir = '/project/cq-training-1/project2/teams/team12/data/'
    best_model_path = 'saved_model/Transformer-num_layers_2-d_model_128-num_heads_8-dff_512_fr_to_en_False_embedding_None_embedding_dim_128_back_translation_True_ratio_4.0'
    path_en = os.path.join(data_dir, 'train.lang1')
    path_fr = os.path.join(data_dir, 'train.lang2')
    
    # Create vocabs
    logger.info('Creating vocab...')
    word2idx_en, idx2word_en = utils.create_vocab(path_en, vocab_size=None)
    word2idx_fr, idx2word_fr = utils.create_vocab(path_fr, vocab_size=None)

    # Load data
    logger.info('Loading data...')
    data = utils.load_data(input_file_path, word2idx_en)
    dataset = tf.data.Dataset.from_generator(lambda: [ex for ex in data],
                                                tf.int64,
                                                output_shapes=tf.TensorShape([None])).padded_batch(
                                                    128, padded_shapes=[None])
    # Load model
    model_config = {'num_layers': 2, 'd_model': 128, 'dff': 512, 'num_heads': 8}
    model = Transformer(model_config, len(word2idx_en), word2idx_fr)
    model.load_weights(os.path.join(best_model_path, "model"))

    # Write prediction to file
    with open(pred_file_path, 'w') as f:
        logger.info('Opening file and writing predictions...')
        for batch in tqdm(dataset, desc='Translating...', total=len(data) // 128 + 1):
            preds = model({'inputs': batch, 'labels': tf.zeros_like(batch)})
            for pred in preds:
                sentence = utils.generate_sentence_from_probabilities(pred.numpy(), idx2word_fr)
                f.writelines([sentence, '\n'])
    ##### MODIFY ABOVE #####


def compute_bleu(pred_file_path: str, target_file_path: str, print_all_scores: bool):
    """

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).
        print_all_scores: if True, will print one score per example.

    Returns: None

    """
    out = subprocess.run(["sacrebleu", "--input", pred_file_path, target_file_path, '--tokenize',
                          'none', '--sentence-level', '--score-only'],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = out.stdout.split('\n')
    if print_all_scores:
        print('\n'.join(lines[:-1]))
    else:
        scores = [float(x) for x in lines[:-1]]
        print('final avg bleu score: {:.2f}'.format(sum(scores) / len(scores)))


def main():
    parser = argparse.ArgumentParser('script for evaluating a model.')
    parser.add_argument('--target-file-path', help='path to target (reference) file', required=True)
    parser.add_argument('--input-file-path', help='path to input file', required=True)
    parser.add_argument('--print-all-scores', help='will print one score per sentence',
                        action='store_true')
    parser.add_argument('--do-not-run-model',
                        help='will use --input-file-path as predictions, instead of running the '
                             'model on it',
                        action='store_true')

    args = parser.parse_args()

    if args.do_not_run_model:
        compute_bleu(args.input_file_path, args.target_file_path, args.print_all_scores)
    else:
        _, pred_file_path = tempfile.mkstemp()
        generate_predictions(args.input_file_path, pred_file_path)
        compute_bleu(pred_file_path, args.target_file_path, args.print_all_scores)


if __name__ == '__main__':
    main()