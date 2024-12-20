import copy

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.nets_utils import lengths_list_to_bool, pad_list, th_accuracy
from espnet.nets.pytorch_backend.transformer.encoder_layer import DropPath
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import InstanceNorm, LayerNorm
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import EMA, set_requires_grad


class USRSingle(nn.Module):
    def __init__(self, cfg, backbone_args=None, pred_other=None):
        super().__init__()
        self.cfg = cfg

        self.odim = 1049
        self.ignore_id = -1

        self.backbone = E2E(self.odim, backbone_args)
        self.predictor_other = instantiate(pred_other) if pred_other else None
        self.target_backbone = self.get_target_model(self.backbone)
        self.out_layer_unlabelled_v = nn.Linear(backbone_args.ddim, self.odim)
        self.out_layer_unlabelled_a = nn.Linear(backbone_args.ddim, self.odim)
        self.out_layer_unlabelled_av = nn.Linear(backbone_args.ddim, self.odim)
        self.out_layer_unlabelled_ctc_v = nn.Linear(backbone_args.adim, self.odim)
        self.out_layer_unlabelled_ctc_a = nn.Linear(backbone_args.adim, self.odim)
        self.out_layer_unlabelled_ctc_av = nn.Linear(backbone_args.adim, self.odim)

        self.d_idim = backbone_args.d_idim
        self.ema = EMA()            
        self.target_dropout_off = cfg.model.target_dropout_off

        self.layer_norm = LayerNorm(backbone_args.adim)
        self.instance_norm = InstanceNorm(backbone_args.adim)

        self.eos = self.odim - 1

    def update_moving_average(self, momentum):
        self.ema.update_moving_average(self.target_backbone, self.backbone, momentum)

    def get_target_model(self, model):
        target_model = copy.deepcopy(model)
        set_requires_grad(target_model, False)
        return target_model

    def set_dropout_mode(self, model, train_mode):
        for m in model.modules():
            if isinstance(m, (DropPath, nn.Dropout)):
                if train_mode:
                    m.train()
                else:
                    m.eval()

    @torch.no_grad()
    def get_target_features(self, x_v, x_a, padding_mask):
        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone.encoder, train_mode=False) 

        e = self.target_backbone.encoder.forward_single(x_v, x_a, padding_mask, return_feats=True)[0]

        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone.encoder, train_mode=True)

        return e

    @torch.no_grad()
    def get_encoder_targets(self, e):
        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone.ctc_av, train_mode=False) 

        ctc_out = self.target_backbone.ctc_av.ctc_lo(e)

        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone.ctc_av, train_mode=True)
        
        return ctc_out

    @torch.no_grad()
    def get_decoder_targets(self, x, padding_mask):
        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone.decoder, train_mode=False)

        ys_in = torch.zeros((len(x), 1, self.d_idim), dtype=x.dtype, device=x.device)
        min_val = float(torch.finfo(ys_in.dtype).min)
        max_val = float(torch.finfo(ys_in.dtype).max)
        ys_in.fill_(min_val)
        ys_in[:, :, -1] = max_val

        cache = None

        out = [None] * len(x)
        idcs = torch.tensor(list(range(len(x)))).to(x.device)

        for _ in range(x.size(1) // 5):
            ys_mask = torch.stack([subsequent_mask(ys_in.size(1), device=x.device)] * len(x))

            ys_out, cache = self.target_backbone.decoder.forward_one_step(
                ys_in if self.cfg.model.soft_inputs else ys_in.argmax(-1), 
                ys_mask,
                x, 
                padding_mask,
                cache=cache, 
            )

            ys_out = self.target_backbone.decoder.out_layer_av(ys_out)
            ys_in = torch.cat([ys_in, ys_out.unsqueeze(1)], dim=1)

            is_eos = ys_out.argmax(dim=-1) == self.eos
            ended_idcs = torch.nonzero(is_eos, as_tuple=False).view(-1).to(x.device)
            remain_idcs = torch.nonzero(is_eos == 0, as_tuple=False).view(-1).to(x.device)
            for i in ended_idcs:
                i = i.item()
                out[idcs[i]] = ys_in[i][1:]
            
            idcs = idcs[remain_idcs]
            ys_in = ys_in[remain_idcs]
            x = x[remain_idcs]
            padding_mask = padding_mask[remain_idcs]
            cache = [c[remain_idcs] for c in cache]

            if not len(idcs):
                break

        for i, idx in enumerate(idcs):
            idx = idx.item()
            out[idx] = ys_in[i][1:]
                
        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone.decoder, train_mode=True)

        return pad_list(out, min_val), lengths_list_to_bool(out)
    
    def get_encoded_features(self, video, audio, padding_mask, return_feats=False):
        x_v, x_a, x_av, _, _ = self.backbone.encoder(video, audio, padding_mask, return_feats=return_feats)
        return x_v, x_a, x_av
    
    def get_encoder_losses(
            self, x_v, x_a, x_av, padding_mask, ctc_targets, is_labelled, mask_conf=None, mask_conf_a=None
        ):
        if is_labelled:
            loss_ctc_v = self.backbone.ctc_v(x_v, padding_mask.sum(-1).squeeze(-1), ctc_targets)
            loss_ctc_a = self.backbone.ctc_a(x_a, padding_mask.sum(-1).squeeze(-1), ctc_targets)
            loss_ctc_av = self.backbone.ctc_av(x_av, padding_mask.sum(-1).squeeze(-1), ctc_targets)
        else:
            if ctc_targets is not None:
                pred_ctc_v = self.out_layer_unlabelled_ctc_v(x_v)
                loss_ctc_v = self.backbone.criterion_ctc(
                    pred_ctc_v, ctc_targets.argmax(-1), torch.logical_or(~mask_conf, ~padding_mask.squeeze(-2))
                )
                pred_ctc_a = self.out_layer_unlabelled_ctc_a(x_a)
                loss_ctc_a = self.backbone.criterion_ctc(
                    pred_ctc_a, ctc_targets.argmax(-1), torch.logical_or(~mask_conf_a, ~padding_mask.squeeze(-2))
                )
                pred_ctc_av = self.out_layer_unlabelled_ctc_av(x_av)
                loss_ctc_av = self.backbone.criterion_ctc(
                    pred_ctc_av, ctc_targets.argmax(-1), torch.logical_or(~mask_conf_a, ~padding_mask.squeeze(-2))
                )
            else:
                loss_ctc_v = loss_ctc_a = loss_ctc_av = None

        return loss_ctc_v, loss_ctc_a, loss_ctc_av
    
    def get_decoder_losses(
            self, 
            x_v,
            x_a, 
            x_av,
            padding_mask, 
            labels, 
            is_labelled, 
            mask_conf=None,
            mask_conf_a=None,
            mask_targets=None,
        ):
        if is_labelled:
            loss_att_v, loss_att_a, loss_att_av, acc_v, acc_a, acc_av = self.backbone.forward_labelled(
                x_v, x_a, x_av, padding_mask, labels
            )
        else:
            e_v, e_a, e_av = self.backbone.forward_unlabelled(x_v, x_a, x_av, padding_mask, labels)
            
            pred_v = self.out_layer_unlabelled_v(e_v)
            loss_att_v = self.backbone.criterion_u(pred_v, labels.argmax(-1), torch.logical_or(~mask_conf, ~mask_targets))
            acc_v = th_accuracy(
                    pred_v.view(-1, self.odim), labels.argmax(-1), ignore_label=self.ignore_id
                )

            pred_a = self.out_layer_unlabelled_a(e_a)
            loss_att_a = self.backbone.criterion_u(pred_a, labels.argmax(-1), torch.logical_or(~mask_conf_a, ~mask_targets))
            acc_a = th_accuracy(
                    pred_a.view(-1, self.odim), labels.argmax(-1), ignore_label=self.ignore_id
                )

            pred_av = self.out_layer_unlabelled_av(e_av)
            loss_att_av = self.backbone.criterion_u(pred_av, labels.argmax(-1), torch.logical_or(~mask_conf_a, ~mask_targets))
            acc_av = th_accuracy(
                    pred_av.view(-1, self.odim), labels.argmax(-1), ignore_label=self.ignore_id
                )

        return loss_att_v, loss_att_a, loss_att_av, acc_v, acc_a, acc_av


class USR(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        pred_v2a = cfg.model.predictor_2a if cfg.model.v2a_weight else None
        self.model = USRSingle(cfg, cfg.model.backbone, pred_v2a)
        self.cfg = cfg

    def update_moving_average(self, momentum):
        self.model.update_moving_average(momentum)