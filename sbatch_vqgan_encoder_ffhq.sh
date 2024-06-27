#!/bin/bash
#SBATCH -A sayandebroy.csmi
#SBATCH -c 30
#SBATCH --gres=gpu:4
#SBATCH --nodelist gnode048
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --output=output_vqgan_encoder_init_63_ffhq_intensity_100__beta_min_10___4__beta_max_0_02__n_steps_1000.txt
#SBATCH --mail-type=ALL


conda init bash
source activate taming

which python


python main.py --base configs/faceshq_vqgan.yaml --init_from logs/2024-06-09T18-19-47_faceshq_vqgan_epoch100/checkpoints/last.ckpt -t True --gpus 0,1,2,3
