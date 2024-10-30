#!/bin/bash
#SBATCH --job-name=usr
#SBATCH --partition={}
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --nodes=4
#SBATCH --time=00:00:00
#SBATCH --account all
#SBATCH --no-requeue

srun python main.py \
    experiment_name=base_low_resource_lrs3_semi \
    trainer.num_nodes=4 \
    optimizer.warmup_epochs=20 \
    trainer.max_epochs=75 \
    optimizer.base_lr=3e-3 \
    model.unlab_rel_weight_v=0.8 \
    model.v_rel_weight=0.3 \
    data.frames_per_gpu=2400 \
    model.backbone.drop_path=0.1 \
    checkpoint.dirpath={} \
    model.transfer_only_encoder=True \
    data.dataset.train_csv={} \
    data.dataset.train_labelled_csv={} \
    data.dataset.val_csv={} \
    data.dataset.test_csv={} \
    data.lrs3_video_dir={} \
    data.lrs3_audio_dir={} \
    model/backbone=resnet_transformer_base \
    model.pretrained_model_path={}