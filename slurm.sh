#!/bin/bash
#SBATCH --gres=gpu:k80:1
#SBATCH --time=12:00:00

module load python/3.7
source ~/venv/bin/activate

cd project2/low-resource-translation
python runs.py --model_name=transformer
