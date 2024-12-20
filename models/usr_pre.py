import copy

from espnet.nets.pytorch_backend.e2e_asr_transformer_pre import E2E
from espnet.nets.pytorch_backend.nets_utils import lengths_list_to_bool, pad_list, th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_distil
from espnet.nets.pytorch_backend.transformer.encoder_layer import DropPath
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import InstanceNorm, LayerNorm
from hydra.utils import instantiate
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import EMA, set_requires_grad


class USRSingle(nn.Module):
    def __init__(self, cfg, backbone_args=None, pred=None):
        super().__init__()
        self.cfg = cfg

        self.odim = 1049
        self.ignore_id = -1

        self.backbone = E2E(self.odim, backbone_args, True)

        self.predictor = instantiate(pred) if pred else None

        self.target_backbone = self.get_target_model(self.backbone.encoder)
        self.out_layer_unlabelled_v = nn.Linear(backbone_args.ddim, self.odim)
        self.out_layer_unlabelled_a = nn.Linear(backbone_args.ddim, self.odim)
        self.out_layer_unlabelled_av = nn.Linear(backbone_args.ddim, self.odim)
        self.out_layer_unlabelled_ctc_v = nn.Linear(backbone_args.adim, self.odim)
        self.out_layer_unlabelled_ctc_a = nn.Linear(backbone_args.adim, self.odim)
        self.out_layer_unlabelled_ctc_av = nn.Linear(backbone_args.adim, self.odim)

        self.d_idim = backbone_args.d_idim
        self.ema = EMA()            
        self.target_dropout_off = cfg.model.target_dropout_off

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
    def get_raven_targets(self, x_v, x_a, padding_mask):
        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone, train_mode=False) 

        e = self.target_backbone.forward_raven(x_v, x_a, padding_mask, avg_feats=self.cfg.model.avg_feats)[0]

        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone, train_mode=True)

        return e

    @torch.no_grad()
    def get_encoder_targets(self, e):
        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone.ctc_av, train_mode=False) 

        ctc_out = self.target_backbone.ctc_av.ctc_lo(e)

        if self.target_dropout_off:
            self.set_dropout_mode(self.target_backbone.ctc_av, train_mode=True)
        
        return ctc_out
    
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

    def get_raven_losses(self, x_v, x_a, x_av, targets, padding_mask, mask):
        mask_loss = torch.cat([mask, mask, mask])
        mask_cat = torch.cat([mask, mask])
        padding_mask_cat = torch.cat([padding_mask, padding_mask])
        targets = torch.cat([targets, targets, targets])

        x = torch.cat([x_v, x_a, x_av])

        p_v = self.predictor(x_v, padding_mask.unsqueeze(-2), token_mask=mask)[0]
        p_a = self.predictor(torch.cat([x_a, x_av]), padding_mask_cat.unsqueeze(-2), token_mask=mask_cat)[0]

        loss = -F.cosine_similarity(torch.cat([p_v, p_a]), targets, dim=-1)
        loss = loss.masked_fill(mask_loss == 0, 0.)
        loss_v, loss_a, loss_av = loss[:len(x_v)], loss[len(x_v):2*len(x_v)], loss[2*len(x_v):]

        return loss_v.sum() / mask.sum(), loss_a.sum() / mask.sum(), loss_av.sum() / mask.sum()


class USR(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        pred = cfg.model.predictor_v2a
        self.model = USRSingle(cfg, cfg.model.backbone, pred)
        self.cfg = cfg

    def update_moving_average(self, momentum):
        self.model.update_moving_average(momentum)