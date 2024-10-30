import os

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


def cut_or_pad(data, size, dim=0):
    # Pad with zeros on the right if data is too short
    # assert abs(data.size(dim) - size) < 2000 
    if data.size(dim) < size:
        # assert False
        padding = size - data.size(dim)
        data = torch.from_numpy(np.pad(data, (0, padding), "constant"))
    # Cut from the right if data is too long
    elif data.size(dim) > size:
        data = data[:size]
    # Keep if data is exactly right
    assert data.size(dim) == size
    return data


class AVDataset(Dataset):
    def __init__(
            self, 
            data_path,
            video_path_prefix_lrs2,
            audio_path_prefix_lrs2,
            video_path_prefix_lrs3, 
            audio_path_prefix_lrs3, 
            video_path_prefix_vox2=None, 
            audio_path_prefix_vox2=None, 
            transforms=None,
            skip_fails=True,
        ):

        self.data_path = data_path
        self.video_path_prefix_lrs3 = video_path_prefix_lrs3
        self.audio_path_prefix_lrs3 = audio_path_prefix_lrs3
        self.video_path_prefix_vox2 = video_path_prefix_vox2
        self.audio_path_prefix_vox2 = audio_path_prefix_vox2
        self.video_path_prefix_lrs2 = video_path_prefix_lrs2
        self.audio_path_prefix_lrs2 = audio_path_prefix_lrs2
        self.transforms = transforms

        self.paths_counts_labels = self.configure_files()
        self.num_fails = 0

        self.skip_fails = skip_fails
    
    def configure_files(self):
        # from https://github.com/facebookresearch/pytorchvideo/blob/874d27cb55b9d7e9df6cd0881e2d7fe9f262532b/pytorchvideo/data/labeled_video_paths.py#L37
        paths_counts_labels = []
        with open(self.data_path, "r") as f:
            for path_count_label in f.read().splitlines():
                tag, file_path, count, label = path_count_label.split(",")
                paths_counts_labels.append((tag, file_path, int(count), [int(lab) for lab in label.split()]))
        return paths_counts_labels

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()
        if not frames:
            print(path)
            return None
        frames = torch.from_numpy(np.stack(frames))
        frames = frames.permute((3, 0, 1, 2))  # TxHxWxC -> # CxTxHxW
        return frames
    
    def load_audio(self, path):
        audio, sr = torchaudio.load(path, normalize=True)
        # assert sr == 16_000
        return audio
        
    def __len__(self):
        return len(self.paths_counts_labels)

    def __getitem__(self, index):
        tag, file_path, count, label = self.paths_counts_labels[index]
        self.video_path_prefix = getattr(self, f"video_path_prefix_{tag}", "")
        self.audio_path_prefix = getattr(self, f"audio_path_prefix_{tag}", "")

        video = self.load_video(os.path.join(self.video_path_prefix, file_path))
        if video is None:
            # raise ValueError(os.path.join(self.video_path_prefix, file_path))
            self.num_fails += 1
            if self.num_fails == 300:
                raise ValueError("Too many file errors.")
            # if count > 450:
            # return self.__getitem__(index + 1)
            if self.skip_fails:
                return {'video': None, 'video_aug': None, 'audio': None, 'audio_aug': None, 'label': None, 'path': None}
            else:
                return self.__getitem__(index + 1)
        
        if tag == "wild":
            audio_clean = audio_aug = None
        else:
            audio = self.load_audio(os.path.join(self.audio_path_prefix, file_path[:-4] + ".wav"))
            audio = cut_or_pad(audio.squeeze(0), video.size(1) * 640)
            audio_clean = self.transforms['audio'](audio.unsqueeze(0)).squeeze(0)
            audio_aug = self.transforms['audio_aug'](audio.unsqueeze(0)).squeeze(0)

        video_clean = self.transforms['video'](video).permute((1, 2, 3, 0))
        video_aug = self.transforms['video_aug'](video).permute((1, 2, 3, 0))

        return {
            'video': video_clean, 
            'video_aug': video_aug, 
            'audio': audio_clean, 
            'audio_aug': audio_aug, 
            'label': torch.tensor(label),
            'path': file_path,
        }
