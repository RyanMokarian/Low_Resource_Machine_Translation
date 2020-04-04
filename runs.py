import os

import fire


def main():

    seq2seqgru_configs = [{
        'embedding_dim': 64,
        'encoder_units': 128,
        'decoder_units': 128,
        'n_layers': 1
    }, {
        'embedding_dim': 128,
        'encoder_units': 256,
        'decoder_units': 256,
        'n_layers': 1
    }, {
        'embedding_dim': 256,
        'encoder_units': 512,
        'decoder_units': 512,
        'n_layers': 1
    }, {
        'embedding_dim': 64,
        'encoder_units': 128,
        'decoder_units': 128,
        'n_layers': 2
    }, {
        'embedding_dim': 128,
        'encoder_units': 256,
        'decoder_units': 256,
        'n_layers': 2
    }, {
        'embedding_dim': 64,
        'encoder_units': 128,
        'decoder_units': 128,
        'n_layers': 3
    }]

    for config in seq2seqgru_configs:
        os.system(
            f'python train.py --batch_size=25 --model_config={{embedding_dim:{config["embedding_dim"]}\,'\
                                                             f'encoder_units:{config["encoder_units"]}\,'\
                                                             f'decoder_units:{config["decoder_units"]}\,'\
                                                             f'n_layers:{config["n_layers"]}}}'
        )


if __name__ == "__main__":
    fire.Fire(main)