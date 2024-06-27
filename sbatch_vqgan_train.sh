#!/bin/bash
#SBATCH -A sayandebroy.csmi
#SBATCH -c 10
#SBATCH --gres=gpu:4
#SBATCH --nodelist gnode081
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=output_vqgan_ffhd_train.txt
#SBATCH --mail-type=ALL



conda init bash
source activate taming

which python

python main.py --base configs/faceshq_vqgan.yaml -t True --gpus 0,1,2,3
