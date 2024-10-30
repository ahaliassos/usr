#!/bin/bash
#SBATCH --job-name=usr
#SBATCH --partition=...
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --time=00:00:00
#SBATCH --account all
#SBATCH --no-requeue

srun python main_ft.py \
    experiment_name=base_high_resource_lrs3 \
    test=True \
    data.dataset.test_csv={} \
    data.lrs3_video_dir={} \
    data.lrs3_audio_dir={} \
    model/backbone=resnet_transformer_base \
    model.pretrained_model_path={}