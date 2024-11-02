from copy import deepcopy
import os

from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from .dataset import AVDataset
from .samplers import ByFrameCountSampler, DistributedSamplerWrapper, RandomSamplerWrapper
from .transforms import AdaptiveLengthTimeMask, NormalizeVideo


def pad(samples, pad_val=0.0):
    lengths = [len(s) for s in samples]
    max_size = max(lengths)
    sample_shape = list(samples[0].shape[1:])
    collated_batch = samples[0].new_zeros([len(samples), max_size] + sample_shape)
    for i, sample in enumerate(samples):
        diff = len(sample) - max_size
        if diff == 0:
            collated_batch[i] = sample
        else:
            collated_batch[i] = torch.cat(
                [sample, sample.new_full([-diff] + sample_shape, pad_val)]
            )
    if len(samples[0].shape) < 3:
        collated_batch = collated_batch.unsqueeze(1)
    else:
        collated_batch = collated_batch.permute((0, 4, 1, 2, 3)) # [B, T, H, W, C] -> [B, C, T, H, W]
    return collated_batch, lengths


def collate_pad(batch):
    batch_out = {}
    for data_type in ('video', 'video_aug', 'audio', 'audio_aug', 'label'):
        pad_val = -1 if data_type == 'label' else 0.0
        c_batch, sample_lengths = pad([s[data_type] for s in batch if s[data_type] is not None], pad_val)
        batch_out[data_type] = c_batch
        batch_out[data_type + '_lengths'] = sample_lengths
    
    batch_out["path"] = [s["path"] for s in batch if s["path"] is not None]
        
    return batch_out


class DataModule(LightningDataModule):

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.total_gpus = self.cfg.gpus * self.cfg.trainer.num_nodes
        print('total gpus:', self.total_gpus)

    def _video_transform(self, mode):
        args = self.cfg.data
        transform = [
            Lambda(lambda x: x / 255.),
        ] + (
            [
                RandomCrop(args.crop_type.random_crop_dim),
                Resize(args.crop_type.resize_dim),
                RandomHorizontalFlip(args.horizontal_flip_prob)
            ]
            if mode == "train" else [CenterCrop(args.crop_type.random_crop_dim), Resize(args.crop_type.resize_dim)]
        )
        if self.cfg.data.channel.in_video_channels == 1:
            transform.extend([Lambda(lambda x: x.transpose(0, 1)), Grayscale(), Lambda(lambda x: x.transpose(0, 1))])
        transform.append(NormalizeVideo(args.channel.obj.mean, args.channel.obj.std))

        transform_aug = deepcopy(transform)
        if mode == "train":
            transform_aug.append(
                AdaptiveLengthTimeMask(
                    window=int(args.timemask_window * 25),
                    stride=int(args.timemask_stride * 25),
                    replace_with_zero=True
                )
            )


        # return Compose(transform)
        return Compose(transform), Compose(transform_aug)

    def _raw_audio_transform(self, mode):
        args = self.cfg.data
        transform = [Lambda(lambda x: x)]
        transform_aug = deepcopy(transform)
        if mode == "train":
            transform_aug.append(
                AdaptiveLengthTimeMask(
                    window=int(args.timemask_window_audio * 16_000),
                    stride=int(args.timemask_stride_audio * 16_000),
                    replace_with_zero=True
                )
            )

        # return Compose(transform)
        return Compose(transform), Compose(transform_aug)

    def _dataloader(self, ds, sampler, collate_fn):
        return DataLoader(
            ds,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            batch_sampler=sampler,
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        ds_args = self.cfg.data.dataset

        transform_video, transform_video_aug = self._video_transform(mode='train')
        transform_audio, transform_audio_aug = self._raw_audio_transform(mode='train')
        
        train_ds_unlabel = AVDataset(
            data_path=ds_args.train_csv,
            video_path_prefix_lrs2=self.cfg.data.lrs2_video_dir,
            audio_path_prefix_lrs2=self.cfg.data.lrs2_audio_dir,
            video_path_prefix_lrs3=self.cfg.data.lrs3_video_dir,
            audio_path_prefix_lrs3=self.cfg.data.lrs3_audio_dir,
            video_path_prefix_vox2=self.cfg.data.vox2_video_dir,
            audio_path_prefix_vox2=self.cfg.data.vox2_audio_dir,
            transforms={
                'video': transform_video, 'video_aug': transform_video_aug, 'audio': transform_audio, 'audio_aug': transform_audio_aug
            },
        )

        sampler_unlabel = ByFrameCountSampler(train_ds_unlabel, self.cfg.data.frames_per_gpu)
        if self.total_gpus > 1:
            sampler_unlabel = DistributedSamplerWrapper(sampler_unlabel)
        else:
            sampler_unlabel = RandomSamplerWrapper(sampler_unlabel)
        dataloader_unlabel = self._dataloader(train_ds_unlabel, sampler_unlabel, collate_pad)
        return dataloader_unlabel

    def val_dataloader(self):
        ds_args = self.cfg.data.dataset

        transform_video, transform_video_aug = self._video_transform(mode='val')
        transform_audio, transform_audio_aug = self._raw_audio_transform(mode='val')

        val_ds = AVDataset(
            data_path=ds_args.val_csv,
            video_path_prefix_lrs2=self.cfg.data.lrs2_video_dir,
            audio_path_prefix_lrs2=self.cfg.data.lrs2_audio_dir,
            video_path_prefix_lrs3=self.cfg.data.lrs3_video_dir,
            audio_path_prefix_lrs3=self.cfg.data.lrs3_audio_dir,
            transforms={
                'video': transform_video, 'video_aug': transform_video_aug, 'audio': transform_audio, 'audio_aug': transform_audio_aug
            },
        )
        sampler = ByFrameCountSampler(val_ds, self.cfg.data.frames_per_gpu_val, shuffle=False)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(val_ds, sampler, collate_pad)

    def test_dataloader(self):
        ds_args = self.cfg.data.dataset

        transform_video, transform_video_aug = self._video_transform(mode='val')
        transform_audio, transform_audio_aug = self._raw_audio_transform(mode='val')

        test_ds = AVDataset(
            data_path=ds_args.test_csv,
            video_path_prefix_lrs2=self.cfg.data.lrs2_video_dir,
            audio_path_prefix_lrs2=self.cfg.data.lrs2_audio_dir,
            video_path_prefix_lrs3=self.cfg.data.lrs3_video_dir,
            audio_path_prefix_lrs3=self.cfg.data.lrs3_audio_dir,
            transforms={
                'video': transform_video, 'video_aug': transform_video_aug, 'audio': transform_audio, 'audio_aug': transform_audio_aug
            },
        )
        sampler = ByFrameCountSampler(test_ds, self.cfg.data.frames_per_gpu_val, shuffle=False)
        if self.total_gpus > 1:
            sampler = DistributedSamplerWrapper(sampler, shuffle=False, drop_last=True)
        return self._dataloader(test_ds, sampler, collate_pad)
