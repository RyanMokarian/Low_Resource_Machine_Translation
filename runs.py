import os

import fire

seq2seqgru_configs = [
    {
        'embedding_dim': 64,
        'encoder_units': 128,
        'decoder_units': 128,
        'n_layers': 1
    },
    {
        'embedding_dim': 128,
        'encoder_units': 256,
        'decoder_units': 256,
        'n_layers': 1
    },
    {  # Best params to date
        'embedding_dim': 256,
        'encoder_units': 512,
        'decoder_units': 512,
        'n_layers': 1
    },
    {
        'embedding_dim': 64,
        'encoder_units': 128,
        'decoder_units': 128,
        'n_layers': 2
    },
    {
        'embedding_dim': 128,
        'encoder_units': 256,
        'decoder_units': 256,
        'n_layers': 2
    },
    {
        'embedding_dim': 64,
        'encoder_units': 128,
        'decoder_units': 128,
        'n_layers': 3
    }
]

transformer_configs = [
    {  # Default params
        'num_layers': 4,  # Number of layers
        'd_model': 128,  # Dimension of the embedding
        'dff': 512,  # Dimension of the fully connected layers
        'num_heads': 8  # Number of attention heads
    },
    {
        'num_layers': 4,
        'd_model': 128,
        'dff': 512,
        'num_heads': 4  # Smaller attention, since attention is hard to train with small dataset
    },
    {
        'num_layers': 4,
        'd_model': 128,
        'dff': 256,  # Smaller fully connected layers
        'num_heads': 8
    },
    {
        'num_layers': 2,  # Less layers
        'd_model': 128,
        'dff': 512,
        'num_heads': 8
    },
    {
        'num_layers': 4,
        'd_model': 256,  # Bigger embeddings
        'dff': 512,
        'num_heads': 8
    }
]


def main(model_name: str = 'seq2seqgru'):

    if model_name == 'seq2seqgru':
        for config in seq2seqgru_configs:
            os.system(
                f'python train.py --batch_size=32 --model_config={{embedding_dim:{config["embedding_dim"]}\,'\
                                                                f'encoder_units:{config["encoder_units"]}\,'\
                                                                f'decoder_units:{config["decoder_units"]}\,'\
                                                                f'n_layers:{config["n_layers"]}}}'
            )
    elif model_name == 'transformer':
        for config in transformer_configs:
            os.system(
                f'python train.py --model_config={{num_layers:{config["num_layers"]}\,'\
                                                 f'd_model:{config["d_model"]}\,'\
                                                 f'dff:{config["dff"]}\,'\
                                                 f'num_heads:{config["num_heads"]}}} '\
                                 '--batch_size=128 --epochs=100 --model_name=transformer'
            )


if __name__ == "__main__":
    fire.Fire(main)
