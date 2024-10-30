#!/bin/bash
#SBATCH --job-name=usr
#SBATCH --partition={}
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=16
#SBATCH --time=00:00:00
#SBATCH --account all
#SBATCH --no-requeue

srun python main.py \
    experiment_name=large_low_resource_lrs3vox2_semi \
    data/dataset=vox2 \
    trainer.num_nodes=16 \
    optimizer.warmup_epochs=20 \
    trainer.max_epochs=75 \
    optimizer.base_lr=2e-3 \
    model.unlab_rel_weight_v=0.95 \
    model.unlab_rel_weight_a=0.7 \
    model.v_rel_weight=0.3 \
    data.frames_per_gpu=1400 \
    data.frames_per_gpu_labelled=200 \
    model.backbone.drop_path=0.3 \
    checkpoint.dirpath={} \
    model.transfer_only_encoder=True \
    data.dataset.train_csv={} \
    data.dataset.train_labelled_csv={} \
    data.dataset.val_csv={} \
    data.dataset.test_csv={} \
    data.lrs3_video_dir={} \
    data.lrs3_audio_dir={} \
    data.vox2_video_dir={} \
    data.vox2_audio_dir={} \
    model/backbone=resnet_transformer_large \
    model.pretrained_model_path={}