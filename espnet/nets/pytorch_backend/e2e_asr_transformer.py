# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import logging

import torch

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_av import Encoder
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    KLMaskedLoss, LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import target_mask, target_mask_cont
from espnet.nets.scorers.ctc import CTCPrefixScorer


class E2E(torch.nn.Module):
    def __init__(self, odim, args, ignore_id=-1, self_train=False):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        # Check the relative positional encoding type
        self.rel_pos_type = getattr(args, "rel_pos_type", None)
        if self.rel_pos_type is None and args.transformer_encoder_attn_layer_type == "rel_mha":
            args.transformer_encoder_attn_layer_type = "legacy_rel_mha"
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )

        self.encoder = Encoder(
            idim=args.idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            encoder_attn_layer_type=args.transformer_encoder_attn_layer_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
            zero_triu=getattr(args, "zero_triu", False),
            a_upsample_ratio=args.a_upsample_ratio,
            relu_type=getattr(args, "relu_type", "swish"),
            layerscale=args.layerscale,
            init_values=args.init_values,
            ff_bn_pre=args.ff_bn_pre,
            post_norm=args.post_norm,
            gamma_zero=args.gamma_zero,
            gamma_init=args.gamma_init,
            mask_init_type=args.mask_init_type,
            drop_path=args.drop_path,
            encoder_stride=args.encoder_stride,
        )

        if args.ctc_rel_weight > 0.0:
            self.ctc_v = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
            self.ctc_a = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
            self.ctc_av = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc_v = self.ctc_a = self.ctc_av = None

        self.transformer_input_layer = args.transformer_input_layer
        self.a_upsample_ratio = args.a_upsample_ratio


        # self.out_layer_v = torch.nn.Linear(args.ddim, odim)
        # self.out_layer_a = torch.nn.Linear(args.ddim, odim)
        # self.out_layer_av = torch.nn.Linear(args.ddim, odim)

        if args.ctc_rel_weight < 1.0:
            self.decoder = Decoder(
                idim=args.d_idim,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                input_layer=args.d_input_layer,
                avg_branches=args.d_avg_branches,
                odim=args.d_odim,
                soft_inputs=args.d_soft_inputs,
                proj_decoder = torch.nn.Linear(args.adim, args.ddim) if args.adim != args.ddim else None
            )
        else:
            self.decoder = None

        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        self.criterion = LabelSmoothingLoss(
            self.odim,
            self.ignore_id,
            args.lsm_weight,
            args.transformer_length_normalized_loss,
        )
        if args.d_soft_inputs:
            self.criterion_u = KLMaskedLoss(
                self.odim,
                self.ignore_id,
                temperature=args.temp_att,
                normalize_length=args.transformer_length_normalized_loss,
            )
        else:
            self.criterion_u = self.criterion
        
        if args.d_soft_inputs:
            self.criterion_ctc = KLMaskedLoss(
                self.odim,
                self.ignore_id,
                temperature=args.temp_ctc,
                normalize_length=args.transformer_length_normalized_loss,
            )
        else:
            self.criterion_ctc = LabelSmoothingLoss(
                self.odim,
                self.ignore_id,
                args.lsm_weight_ctc,
                args.transformer_length_normalized_loss,
            )
        self.args = args

        if self_train:
            self.att_unl_a = torch.nn.Linear(args.adim, 1049)
            self.att_unl_av = torch.nn.Linear(args.adim, 1049)
            self.att_unl_v = torch.nn.Linear(args.adim, 1049)
            
            self.ctc_unl_a = CTC(
                1049, 
                args.adim, 
                args.dropout_rate, 
                ctc_type=args.ctc_type, 
                reduce=True
            )
            self.ctc_unl_av = CTC(
                1049, 
                args.adim, 
                args.dropout_rate, 
                ctc_type=args.ctc_type, 
                reduce=True
            )
            self.ctc_unl_v = CTC(
                1049, 
                args.adim, 
                args.dropout_rate, 
                ctc_type=args.ctc_type, 
                reduce=True
            )

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc_v, self.ctc_a, self.ctc_av, self.eos))  # TESTING NEED TO DO IT FOR BOTH V AND A

    def forward_labelled(self, x_v, x_a, x_av, padding_mask, targets):
        ys_in_pad, ys_out_pad = add_sos_eos(targets, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)

        ys_in_pad = torch.cat([ys_in_pad, ys_in_pad, ys_in_pad])
        ys_mask = torch.cat([ys_mask, ys_mask, ys_mask])
        x = torch.cat([x_v, x_a, x_av])
        padding_mask = torch.cat([padding_mask, padding_mask, padding_mask])

        out = self.decoder(ys_in_pad, ys_mask, x, padding_mask, ignore_soft=True)[0]

        pred_v = self.decoder.out_layer_v(out[:len(x_v)])
        pred_a = self.decoder.out_layer_a(out[len(x_v):2*len(x_v)])
        pred_av = self.decoder.out_layer_av(out[2*len(x_v):])

        # print(pred_av.softmax(-1).max(-1))

        loss_att_v = self.criterion(pred_v, ys_out_pad)
        loss_att_a = self.criterion(pred_a, ys_out_pad)
        loss_att_av = self.criterion(pred_av, ys_out_pad)

        acc_v = th_accuracy(
                pred_v.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        acc_a = th_accuracy(
                pred_a.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        acc_av = th_accuracy(
                pred_av.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        return loss_att_v, loss_att_a, loss_att_av, acc_v, acc_a, acc_av

    
    def forward_unlabelled(self, x_v, x_a, x_av, padding_mask, targets):
        ys_mask = target_mask_cont(targets)
        _sos = torch.zeros((len(x_v), 1, self.args.d_idim), dtype=x_v.dtype, device=x_v.device)
        min_val = float(torch.finfo(_sos.dtype).min)
        max_val = float(torch.finfo(_sos.dtype).max)
        _sos.fill_(min_val)
        _sos[:, :, -1] = max_val
        # _sos = torch.tensor([self.args.d_idim - 1]*len(targets), dtype=targets.dtype, device=targets.device)
        targets = torch.cat([_sos, targets], dim=1)
        if not self.args.d_soft_inputs:
            targets = targets.argmax(-1)
        
        x = torch.cat([x_v, x_a, x_av])
        targets = torch.cat([targets, targets, targets])
        ys_mask = torch.cat([ys_mask, ys_mask, ys_mask])
        padding_mask = torch.cat([padding_mask, padding_mask, padding_mask])

        out = self.decoder(targets[:, :-1], ys_mask, x, padding_mask)[0]
        return out[:len(x_v)], out[len(x_v):2*len(x_v)], out[2*len(x_v):]


    def forward_labelled_ft(
        self, 
        x_v, 
        x_a, 
        x_av, 
        padding_mask, 
        targets
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(targets, self.sos, self.eos, self.ignore_id)
        ys_mask = target_mask(ys_in_pad, self.ignore_id)

        ys_in_pad = torch.cat([ys_in_pad, ys_in_pad, ys_in_pad])
        ys_mask = torch.cat([ys_mask, ys_mask, ys_mask])
        x = torch.cat([x_v, x_a, x_av])
        padding_mask = torch.cat([padding_mask, padding_mask, padding_mask])

        out = self.decoder(ys_in_pad, ys_mask, x, padding_mask, ignore_soft=True)[0]

        pred_v = self.att_unl_v(out[:len(x_v)])
        pred_a = self.att_unl_a(out[len(x_v):2*len(x_v)])
        pred_av = self.att_unl_av(out[2*len(x_v):])

        # print(pred_av.softmax(-1).max(-1))

        loss_att_v = self.criterion(pred_v, ys_out_pad)
        loss_att_a = self.criterion(pred_a, ys_out_pad)
        loss_att_av = self.criterion(pred_av, ys_out_pad)

        acc_v = th_accuracy(
                pred_v.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        acc_a = th_accuracy(
                pred_a.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        acc_av = th_accuracy(
                pred_av.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        return loss_att_v, loss_att_a, loss_att_av, acc_v, acc_a, acc_av