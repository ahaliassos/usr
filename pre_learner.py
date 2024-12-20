from fairseq_manual.data_utils import compute_mask_indices
from hydra.utils import instantiate
import torch
from pytorch_lightning import LightningModule

from schedulers.warmup_cosine import WarmupCosineScheduler

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from models.usr_pre import USR


class SSLLearner(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        if cfg.compile_model:
            self.model =  torch.compile(USR(cfg))
        else:
            self.model = USR(cfg)

        if cfg.debug.log_gradients:
            self.logger.experiment.watch(self.model, log="gradients")
        
        self.raven_targets = None

        self.ignore_id = -1

        self.automatic_optimization = False
    

    def get_mask(self, data, mask_prob, mask_length):
        B, C, T, H, W = data["video"].shape
        mask = ~compute_mask_indices(
            (B, T),
            ~self.padding_mask,
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

        label = data["label"].squeeze(1)

        mask = self.get_mask(data, self.cfg.data.mask_prob_audio, self.cfg.data.mask_length_audio)
        mask_video = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        mask_audio = torch.repeat_interleave(
            mask, data["audio"].size(2) // data["video"].size(2), -1
        )

        video = (data["video"] * mask_video).squeeze(1)
        audio = (data["audio"] * mask_audio.unsqueeze(1)).transpose(1, 2)

        x_v, x_a, x_av = self.model.model.get_encoded_features(video, audio, self.padding_mask.unsqueeze(-2))

        loss_raven_v, loss_raven_a, loss_raven_av = self.model.model.get_raven_losses(
            x_v, x_a, x_av, self.raven_targets, self.padding_mask, ~mask
        )

        self.log("loss_raven_v", loss_raven_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_raven_a", loss_raven_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_raven_av", loss_raven_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)

        loss = self.cfg.model.v_rel_weight*loss_raven_v + (1-self.cfg.model.v_rel_weight)*loss_raven_a + (1-self.cfg.model.v_rel_weight)*loss_raven_av

        loss_ctc_v, loss_ctc_a, loss_ctc_av = self.model.model.get_encoder_losses(
            x_v.detach(),
            x_a.detach(),
            x_av.detach(),
            self.padding_mask.unsqueeze(-2), 
            label,
            True
        )
        self.log("loss_ctc_v_l", loss_ctc_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_ctc_a_l", loss_ctc_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_ctc_av_l", loss_ctc_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
    
        loss += loss_ctc_v + loss_ctc_a + loss_ctc_av

        loss_att_v, loss_att_a, loss_att_av, acc_v, acc_a, acc_av = self.model.model.get_decoder_losses(
            x_v.detach(),
            x_a.detach(),
            x_av.detach(),
            self.padding_mask.unsqueeze(-2), 
            label,
            True, 
        )

        self.log("loss_att_v_l", loss_att_v, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_att_a_l", loss_att_a, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("loss_att_av_l", loss_att_av, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(label), sync_dist=True)
        self.log("acc_v_l", acc_v, on_step=False, on_epoch=True, batch_size=len(label), sync_dist=True)
        self.log("acc_a_l", acc_a, on_step=False, on_epoch=True, batch_size=len(label), sync_dist=True)
        self.log("acc_av_l", acc_av, on_step=False, on_epoch=True, batch_size=len(label), sync_dist=True)

        loss += loss_att_v + loss_att_a + loss_att_av

        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=self.cfg.optimizer.gradient_clip_val, gradient_clip_algorithm="norm")

        opt.step()
        sch.step()

        self.log('monitoring_step', self.trainer.global_step)  # this is to save the last k checkpoints

    def on_train_batch_start(self, data, batch_idx):
        self.padding_mask = make_non_pad_mask(data["video_lengths"]).to(data["video"].device)
        self.raven_targets = self.model.model.get_raven_targets(
            data["video"].squeeze(1), data["audio"].transpose(1, 2), self.padding_mask.unsqueeze(-2) 
        )
    
    def on_train_batch_end(self, *args):
        momentum = self.momentum_scheduler.get_lr(self.trainer.global_step)  # global step takes into account 2 optimizer steps in PL > 1.5
        self.model.update_moving_average(momentum)
        self.log("momentum", momentum, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

    def shared_val_test_step(self, data):
        video, audio, label = data["video"], data["audio"], data["label"]
        padding_mask_v = make_non_pad_mask(data["video_lengths"]).to(data["video"].device).unsqueeze(-2)

        features_v, features_a, features_av = self.model.model.get_encoded_features(video.squeeze(1), audio.transpose(1, 2), padding_mask_v)

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

        acc_video, acc_audio, acc_av = self.model.model.backbone.forward_labelled(
            features_v, features_a, features_av, padding_mask_v, label
        )[-3:]

        self.log("acc_video_val", acc_video, batch_size=len(label), sync_dist=True)
        self.log("acc_audio_val", acc_audio, batch_size=len(label), sync_dist=True)
        self.log("acc_av_val", acc_av, batch_size=len(label), sync_dist=True)

    def validation_step(self, data, batch_idx):
        self.shared_val_test_step(data)
    
    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def configure_optimizers(self):
        def get_param_groups(
                model, lr_encoder, lr_decoder, weight_decay
        ):
            param_groups = []
            encoder, decoder = [], []
            for name, param in model.named_parameters():
                if "decoder" in name:
                    decoder.append(param)
                else:
                    encoder.append(param)
          
            param_groups.append(
                {"name": "encoder", "params": encoder, "lr": lr_encoder, "weight_decay": weight_decay}
            )
            param_groups.append(
                {"name": "decoder", "params": decoder, "lr": lr_decoder, "weight_decay": weight_decay}
            )
            return param_groups
                
        param_groups = get_param_groups(
            self.model.model,
            self.cfg.optimizer.base_lr,
            self.cfg.optimizer.base_lr_decoder,
            self.cfg.optimizer.optim.weight_decay,
        )

        optimizer = instantiate(self.cfg.optimizer.optim.obj, param_groups)

        warmup_epochs = self.cfg.optimizer.warmup_epochs
        train_len = len(self.trainer.datamodule.train_dataloader())
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
