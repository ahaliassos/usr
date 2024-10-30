#!/bin/bash
#SBATCH --job-name=usr
#SBATCH --partition={}
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --time=00:00:00
#SBATCH --account all
#SBATCH --no-requeue

srun python main.py \
    experiment_name=large_high_resource_lrs3vox2 \
    test=True \
    data.dataset.test_csv={} \
    data.lrs3_video_dir={} \
    data.lrs3_audio_dir={} \
    model/backbone=resnet_transformer_large \
    model.pretrained_model_path={}