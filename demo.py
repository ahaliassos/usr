import os

import cv2
import hydra
import numpy as np
import pytorchvideo
import torch
import torchaudio
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
)

from data.transforms import NormalizeVideo
from espnet.asr.asr_utils import add_results_to_json, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.lm.transformer import TransformerLM
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.scorers.length_bonus import LengthBonus
from utils.utils import UNIGRAM1000_LIST


def load_video(path):
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

def load_audio(path):
    audio, sr = torchaudio.load(path, normalize=True)
    return audio


def video_transform():
    transform = [
        Lambda(lambda x: x / 255.), 
        CenterCrop(88), 
        Lambda(lambda x: x.transpose(0, 1)), 
        Grayscale(), 
        Lambda(lambda x: x.transpose(0, 1)),
        NormalizeVideo(mean=(0.421,), std=(0.165,))
    ]

    return Compose(transform)


def get_beam_search(cfg, model):
    token_list = UNIGRAM1000_LIST
    odim = len(token_list)

    scorers = model.scorers()

    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))

    weights = dict(
        decoder=1.0 - cfg.decode.ctc_weight,
        ctc=cfg.decode.ctc_weight,
        lm=cfg.decode.lm_weight,
        length_bonus=cfg.decode.penalty,
    )
    beam_search = BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=odim - 1,
        eos=odim - 1,
        token_list=token_list,
        pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder",
    )

    return beam_search


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.avi")
    audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example.wav")

    video = load_video(video_path)
    audio = load_audio(audio_path)

    video = video_transform()(video)

    model = torch.compile(E2E(1049, cfg.model.backbone))

    ckpt = torch.load(cfg.model.pretrained_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt)

    beam_search = get_beam_search(cfg, model)

    # AV
    feat, _, _ = model.encoder.forward_single(xs_v=video, xs_a=audio.unsqueeze(0).transpose(1, 2))
    
    nbest_hyps = beam_search(
            x=feat.squeeze(0),
            modality="av",
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio
        )
    
    nbest_hyps = [
        h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]
    ]

    transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
    transcription = transcription.replace("<eos>", "")
    transcription = transcription.replace("▁", " ").strip()

    print("AV transcription:", transcription)

    # A
    feat, _, _ = model.encoder.forward_single(xs_a=audio.unsqueeze(0).transpose(1, 2))
    
    nbest_hyps = beam_search(
            x=feat.squeeze(0),
            modality="a",
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio
        )
    
    nbest_hyps = [
        h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]
    ]

    transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
    transcription = transcription.replace("<eos>", "")
    transcription = transcription.replace("▁", " ").strip()

    print("A transcription:", transcription)

    # V
    feat, _, _ = model.encoder.forward_single(xs_v=video)
    
    nbest_hyps = beam_search(
            x=feat.squeeze(0),
            modality="v",
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio
        )
    
    nbest_hyps = [
        h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]
    ]

    transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
    transcription = transcription.replace("<eos>", "")
    transcription = transcription.replace("▁", " ").strip()

    print("V transcription:", transcription)

if __name__ == "__main__":
    main()







