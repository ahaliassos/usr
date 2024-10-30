from fairseq_manual.data_utils import compute_mask_indices
from hydra.utils import instantiate
import torch
from torch.optim import AdamW
from pytorch_lightning import LightningModule
import random

from schedulers.warmup_cosine import WarmupCosineScheduler

from espnet.asr.asr_utils import add_results_to_json, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.lm.transformer import TransformerLM
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.scorers.length_bonus import LengthBonus
from metrics import WER
from utils.utils import ids_to_str, set_requires_grad, UNIGRAM1000_LIST, get_param_groups


class SSLLearner(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        if cfg.compile_model:
            self.model = torch.compile(instantiate(cfg.model.obj, cfg))
        else:
            self.model = instantiate(cfg.model.obj, cfg)

        if cfg.model.pretrained_model_path:
            print("Load pretrained model weights")
            ckpt = torch.load(cfg.model.pretrained_model_path, map_location=lambda storage, loc: storage)

            if cfg.model.transfer_only_encoder:
                ckpt = {k[39:]: v for k, v in ckpt.items() if k.startswith('model._orig_mod.model.backbone.encoder')}
                ckpt = {k: v for k, v in ckpt.items() if not k.startswith("after_norm")}
                self.model.model.backbone.encoder.load_state_dict(ckpt, strict=False)
                self.model.model.target_backbone.encoder.load_state_dict(ckpt, strict=False)
            else:
                self.model.load_state_dict(ckpt)

        if cfg.debug.log_gradients:
            self.logger.experiment.watch(self.model, log="gradients")
        
        self.logits_at_u = self.mask_lengths_v = self.mask_lengths_a = None

        self.ignore_id = -1
        self.beam_search_video = self.get_beam_search(self.model.model.backbone)
        self.beam_search_audio = self.get_beam_search(self.model.model.backbone)
        self.beam_search_av = self.get_beam_search(self.model.model.backbone)
        self.wer_video = WER()
        self.wer_audio = WER()
        self.wer_av = WER()

        self.automatic_optimization = False
    

    def get_beam_search(self, model):
        token_list = UNIGRAM1000_LIST

        odim = len(token_list)
        self.token_list = token_list

        scorers = model.scorers()

        if self.cfg.decode.lm_weight and self.cfg.model.pretrained_lm_path:
            lm = TransformerLM(len(token_list), self.cfg.model.language_model)
            set_requires_grad(lm, False)
            print("Load pretrained language model weights")
            torch_load(self.cfg.model.pretrained_lm_path, lm)
        else:
            lm = None

        scorers["lm"] = lm
        scorers["length_bonus"] = LengthBonus(len(token_list))

        weights = dict(
            decoder=1.0 - self.cfg.decode.ctc_weight,
            ctc=self.cfg.decode.ctc_weight,
            lm=self.cfg.decode.lm_weight,
            length_bonus=self.cfg.decode.penalty,
        )
        beam_search = BatchBeamSearch(
            beam_size=self.cfg.decode.beam_size,
            vocab_size=len(token_list),
            weights=weights,
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=token_list,
            pre_beam_score_key=None if self.cfg.decode.ctc_weight == 1.0 else "decoder",
        )

        return beam_search

    def get_mask(self, data, padding_mask, mask_prob, mask_length):
        B, C, T, H, W = data["video"].shape
        mask = ~compute_mask_indices(
            (B, T),
            ~padding_mask,
            mask_prob,
            mask_length,
            min_masks=1
        )
        return torch.from_numpy(mask).to(data["video"].device)
            
    def training_step(self, data, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        # zero grad
        opt.zero_grad()  

        data_u, data_l = data["unlabelled"], data["labelled"]
        label_l = data_l["label"].squeeze(1)

        video = data_u["video_aug"].squeeze(1)
        audio = data_u["audio_aug"].transpose(1, 2)

        x_v, x_a, x_av = self.model.model.get_encoded_features(video, audio, self.padding_mask_u.unsqueeze(-2), True)

        loss = 0.0
        if self.cfg.model.ctc_rel_weight > 0.0:
            mask_conf_ctc = self.ctc_at_u.softmax(-1).max(-1)[0].ge(self.cfg.model.conf_threshold_ctc)
            mask_conf_ctc_a = self.ctc_at_u.softmax(-1).max(-1)[0].ge(self.cfg.model.conf_threshold_ctc_a)
            self.log(
                "avg_max_prob_ctc_u", 
                self.ctc_at_u.softmax(-1).max(-1)[0].mean(), 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True, 
                batch_size=len(label_l), 
                sync_dist=True
            )
            self.log(
                "avg_above_tresh_ctc_u", mask_conf_ctc.sum() / (mask_conf_ctc.size(0)*mask_conf_ctc.size(1)), 
                on_step=False,
                on_epoch=True, 
                batch_size=len(label_l), 
                sync_dist=True
            )
            
            loss_ctc_v, loss_ctc_a, loss_ctc_av = self.model.model.get_encoder_losses(
                x_v,
                x_a,
                x_av,
                self.padding_mask_u.unsqueeze(-2), 
                self.ctc_at_u,
                False,
                mask_conf_ctc,
                mask_conf_ctc_a,
            )
            self.log(
                "loss_ctc_v_u", loss_ctc_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True
            )
            self.log(
                "loss_ctc_a_u", loss_ctc_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True
            )
            self.log(
                "loss_ctc_av_u", loss_ctc_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True
            )

            loss += self.cfg.model.unlab_rel_weight_v*self.cfg.model.ctc_rel_weight*self.cfg.model.v_rel_weight*loss_ctc_v
            loss += self.cfg.model.unlab_rel_weight_a*self.cfg.model.ctc_rel_weight*(1 - self.cfg.model.v_rel_weight)*loss_ctc_a
            loss += self.cfg.model.unlab_rel_weight_a*self.cfg.model.ctc_rel_weight*(1 - self.cfg.model.v_rel_weight)*loss_ctc_av

        if self.cfg.model.ctc_rel_weight < 1.0:
            mask_conf = self.logits_at_u.softmax(-1).max(-1)[0].ge(self.cfg.model.conf_threshold)
            mask_conf_a = self.logits_at_u.softmax(-1).max(-1)[0].ge(self.cfg.model.conf_threshold_a)
            loss_att_v, loss_att_a, loss_att_av, acc_v, acc_a, acc_av = self.model.model.get_decoder_losses(
                x_v,
                x_a,
                x_av,
                self.padding_mask_u.unsqueeze(-2), 
                self.logits_at_u,
                False, 
                mask_conf,
                mask_conf_a,
                self.mask_lengths_a
            )

            self.log(
                "loss_att_v_u", loss_att_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True
            )
            self.log(
                "loss_att_a_u", loss_att_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True
            )
            self.log(
                "loss_att_av_u", loss_att_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True
            )
            self.log(
                "acc_v_u", acc_v, on_step=False, on_epoch=True, batch_size=len(label_l), sync_dist=True
            )
            self.log(
                "acc_a_u", acc_a, on_step=False, on_epoch=True, batch_size=len(label_l), sync_dist=True
            )
            self.log(
                "acc_av_u", acc_av, on_step=False, on_epoch=True, batch_size=len(label_l), sync_dist=True
            )
            self.log(
                "avg_above_tresh_u", mask_conf.sum() / (mask_conf.size(0)*mask_conf.size(1)), 
                on_step=False, 
                on_epoch=True, 
                batch_size=len(label_l), 
                sync_dist=True
            )
            self.log(
                "avg_above_tresh_u_a", mask_conf_a.sum() / (mask_conf_a.size(0)*mask_conf_a.size(1)), 
                on_step=False, 
                on_epoch=True, 
                batch_size=len(label_l), 
                sync_dist=True
            )
            self.log(
                "avg_max_prob_u", 
                self.logits_at_u.softmax(-1).max(-1)[0].mean(), 
                on_step=True, 
                on_epoch=True, 
                prog_bar=True, 
                batch_size=len(label_l), 
                sync_dist=True
            )

            loss += self.cfg.model.unlab_rel_weight_v*(1-self.cfg.model.ctc_rel_weight)*self.cfg.model.v_rel_weight*loss_att_v
            loss += self.cfg.model.unlab_rel_weight_a*(1-self.cfg.model.ctc_rel_weight)*(1-self.cfg.model.v_rel_weight)*loss_att_a
            loss += self.cfg.model.unlab_rel_weight_a*(1-self.cfg.model.ctc_rel_weight)*(1-self.cfg.model.v_rel_weight)*loss_att_av

        # backward
        self.manual_backward(loss)

        video = data_l["video_aug"].squeeze(1)
        audio = data_l["audio_aug"].transpose(1, 2)

        x_v, x_a, x_av = self.model.model.get_encoded_features(video, audio, self.padding_mask_l.unsqueeze(-2))

        loss = 0.0
        if self.cfg.model.ctc_rel_weight > 0.0:
            loss_ctc_v, loss_ctc_a, loss_ctc_av = self.model.model.get_encoder_losses(
                x_v,
                x_a,
                x_av,
                self.padding_mask_l.unsqueeze(-2), 
                label_l,
                True
            )
            self.log("loss_ctc_v_l", loss_ctc_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True)
            self.log("loss_ctc_a_l", loss_ctc_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True)
            self.log("loss_ctc_av_l", loss_ctc_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True)
        
            loss += (1-self.cfg.model.unlab_rel_weight_v)*self.cfg.model.ctc_rel_weight*self.cfg.model.v_rel_weight*loss_ctc_v 
            loss += (1-self.cfg.model.unlab_rel_weight_a)*self.cfg.model.ctc_rel_weight*(1-self.cfg.model.v_rel_weight)*loss_ctc_a
            loss += (1-self.cfg.model.unlab_rel_weight_a)*self.cfg.model.ctc_rel_weight*(1-self.cfg.model.v_rel_weight)*loss_ctc_av

        if self.cfg.model.ctc_rel_weight < 1.0:
            loss_att_v, loss_att_a, loss_att_av, acc_v, acc_a, acc_av = self.model.model.get_decoder_losses(
                x_v,
                x_a,
                x_av,
                self.padding_mask_l.unsqueeze(-2), 
                label_l,
                True, 
            )

            self.log("loss_att_v_l", loss_att_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True)
            self.log("loss_att_a_l", loss_att_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True)
            self.log("loss_att_av_l", loss_att_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label_l), sync_dist=True)
            self.log("acc_v_l", acc_v, on_step=False, on_epoch=True, batch_size=len(label_l), sync_dist=True)
            self.log("acc_a_l", acc_a, on_step=False, on_epoch=True, batch_size=len(label_l), sync_dist=True)
            self.log("acc_av_l", acc_av, on_step=False, on_epoch=True, batch_size=len(label_l), sync_dist=True)

            loss += (1-self.cfg.model.unlab_rel_weight_v)*(1-self.cfg.model.ctc_rel_weight)*self.cfg.model.v_rel_weight*loss_att_v
            loss += (1-self.cfg.model.unlab_rel_weight_a)*(1-self.cfg.model.ctc_rel_weight)*(1-self.cfg.model.v_rel_weight)*loss_att_a
            loss += (1-self.cfg.model.unlab_rel_weight_a)*(1-self.cfg.model.ctc_rel_weight)*(1-self.cfg.model.v_rel_weight)*loss_att_av

        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=self.cfg.optimizer.gradient_clip_val, gradient_clip_algorithm="norm")
        opt.step()
        sch.step()

        self.log('monitoring_step', self.trainer.global_step)  # this is to save the last k checkpoints

    def on_train_batch_start(self, data, batch_idx):
        data_u, data_l = data["unlabelled"], data["labelled"]
        self.padding_mask_u = make_non_pad_mask(data_u["video_lengths"]).to(data_u["video"].device)
        self.padding_mask_l = make_non_pad_mask(data_l["video_lengths"]).to(data_l["video"].device)
        
        e_a = self.model.model.get_target_features(
            data_u["video"].squeeze(1), data_u["audio"].transpose(1, 2), self.padding_mask_u.unsqueeze(-2) 
        )

        if self.cfg.model.ctc_rel_weight > 0.0:
            self.ctc_at_u = self.model.model.get_encoder_targets(e_a)

        if self.cfg.model.ctc_rel_weight < 1.0:
            self.logits_at_u, self.mask_lengths_a = self.model.model.get_decoder_targets(
                e_a, self.padding_mask_u.unsqueeze(-2)
            )
    
    def on_train_batch_end(self, *args):
        momentum = self.momentum_scheduler.get_lr(self.trainer.global_step)  # global step takes into account 2 optimizer steps in PL > 1.5
        self.model.update_moving_average(momentum)
        self.log("momentum", momentum, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

    def shared_val_test_step(self, data):
        video, audio, label = data["video"], data["audio"], data["label"]
        padding_mask_v = make_non_pad_mask(data["video_lengths"]).to(data["video"].device).unsqueeze(-2)

        features_v, features_a, features_av = self.model.model.get_encoded_features(video.squeeze(1), audio.transpose(1, 2), padding_mask_v)

        if self.cfg.model.ctc_rel_weight > 0.0:
            loss_ctc_v = self.model.model.backbone.ctc_v(
                features_v, torch.tensor(data["video_lengths"], device=features_v.device), data["label"].squeeze(1)
            )
            loss_ctc_a = self.model.model.backbone.ctc_a(
                features_a, torch.tensor(data["video_lengths"], device=features_a.device), data["label"].squeeze(1)
            )
            loss_ctc_av = self.model.model.backbone.ctc_av(
                features_av, torch.tensor(data["video_lengths"], device=features_a.device), data["label"].squeeze(1)
            )

            self.log("loss_ctc_v_val", loss_ctc_v, batch_size=len(label), sync_dist=True)
            self.log("loss_ctc_a_val", loss_ctc_a, batch_size=len(label), sync_dist=True)
            self.log("loss_ctc_av_val", loss_ctc_av, batch_size=len(label), sync_dist=True)

        if self.cfg.model.ctc_rel_weight < 1.0:
            acc_video, acc_audio, acc_av = self.model.model.backbone.forward_labelled(
                features_v, features_a, features_av, padding_mask_v, label
            )[-3:]

            self.log("acc_video_val", acc_video, batch_size=len(label), sync_dist=True)
            self.log("acc_audio_val", acc_audio, batch_size=len(label), sync_dist=True)
            self.log("acc_av_val", acc_av, batch_size=len(label), sync_dist=True)

    def validation_step(self, data, batch_idx):
        self.shared_val_test_step(data)

    def calculate_wer(self, video, audio, padding_mask, labels):
        labels = labels.squeeze(1)
        for vid, aud, label, mask in zip(video, audio, labels, padding_mask):
            feat_v, feat_a, feat_av = self.model.model.get_encoded_features(
                vid.unsqueeze(0), aud.unsqueeze(0), mask.unsqueeze(0).unsqueeze(-2)
            )
            
            nbest_hyps_v = self.beam_search_video(
                    x=feat_v.squeeze(0),
                    modality="v",
                    maxlenratio=self.cfg.decode.maxlenratio,
                    minlenratio=self.cfg.decode.minlenratio
                )
            nbest_hyps_a = self.beam_search_audio(
                    x=feat_a.squeeze(0),
                    modality="a",
                    maxlenratio=self.cfg.decode.maxlenratio,
                    minlenratio=self.cfg.decode.minlenratio
                )
            nbest_hyps_av = self.beam_search_av(
                    x=feat_av.squeeze(0),
                    modality="av",
                    maxlenratio=self.cfg.decode.maxlenratio,
                    minlenratio=self.cfg.decode.minlenratio
                )
            
            nbest_hyps_v = [
                h.asdict() for h in nbest_hyps_v[: min(len(nbest_hyps_v), 1)]
            ]
            nbest_hyps_a = [
                h.asdict() for h in nbest_hyps_a[: min(len(nbest_hyps_a), 1)]
            ]
            nbest_hyps_av = [
                h.asdict() for h in nbest_hyps_av[: min(len(nbest_hyps_av), 1)]
            ]

            transcription_v = add_results_to_json(nbest_hyps_v, self.token_list)
            transcription_v = transcription_v.replace("<eos>", "")
            transcription_a = add_results_to_json(nbest_hyps_a, self.token_list)
            transcription_a = transcription_a.replace("<eos>", "")
            transcription_av = add_results_to_json(nbest_hyps_av, self.token_list)
            transcription_av = transcription_av.replace("<eos>", "")

            label = label[label != self.ignore_id]
            groundtruth = ids_to_str(label, self.token_list)

            groundtruth = groundtruth.replace("▁", " ").strip()
            transcription_v = transcription_v.replace("▁", " ").strip()
            transcription_a = transcription_a.replace("▁", " ").strip()
            transcription_av = transcription_av.replace("▁", " ").strip()

            self.wer_video.update(transcription_v, groundtruth)
            self.wer_audio.update(transcription_a, groundtruth)
            self.wer_av.update(transcription_av, groundtruth)

    def test_step(self, data, batch_idx):
        lengths = torch.tensor(data["video_lengths"], device=data["video"].device)
        padding_mask = make_non_pad_mask(lengths).to(lengths.device)
        self.calculate_wer(
            data["video"].squeeze(1), 
            data["audio"].transpose(1, 2),
            padding_mask, 
            data["label"], 
        )

        print(self.wer_video.compute())
        print(self.wer_audio.compute())
        print(self.wer_av.compute())

    def on_test_epoch_end(self):
        wer_video = self.wer_video.compute()
        wer_audio = self.wer_audio.compute()
        wer_av = self.wer_av.compute()
        print(wer_video)
        print(wer_audio)
        print(wer_av)
        self.log("wer_video", wer_video)
        self.log("wer_audio", wer_audio)
        self.log("wer_av", wer_av)
        self.wer_video.reset()
        self.wer_audio.reset()
        self.wer_av.reset()
    
    def on_train_epoch_start(self):
        for loader in self.trainer.train_dataloader.values():
            sampler = loader.batch_sampler
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def configure_optimizers(self):
        param_groups = get_param_groups(
            self.model.model, 
            self.cfg.model.backbone.elayers, 
            self.cfg.optimizer.base_lr, 
            self.cfg.optimizer.base_lr_other, 
            self.cfg.optimizer.lr_decay_rate,
            min_lr=self.cfg.optimizer.min_lr,
        )

        optimizer = AdamW(
            param_groups, weight_decay=self.cfg.optimizer.weight_decay, betas=self.cfg.optimizer.betas
        )

        warmup_epochs = self.cfg.optimizer.warmup_epochs
        train_len = len(self.trainer.datamodule.train_dataloader()["unlabelled"])
        scheduler = {
            'scheduler': WarmupCosineScheduler(
                optimizer,
                warmup_epochs,
                self.cfg.trainer.max_epochs,
                train_len,
                self.cfg.optimizer.cosine_decay,
                excluded_group=None,
            ),
            'interval': 'step',
            'frequency': 1
        }

        self.momentum_scheduler = instantiate(
            self.cfg.model.momentum_scheduler,
            iter_per_epoch=train_len,
        )

        return [optimizer], [scheduler]

