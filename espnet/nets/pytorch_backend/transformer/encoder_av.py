#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
#from espnet.nets.pytorch_backend.transducer.vgg import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding, # noqa: H301
    MaskEmbedding,
)
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.raw_embeddings import VideoEmbedding
from espnet.nets.pytorch_backend.transformer.raw_embeddings import AudioEmbedding
from espnet.nets.pytorch_backend.backbones.conv3d_extractor  import Conv3dResNet
from espnet.nets.pytorch_backend.backbones.conv1d_extractor  import Conv1dResNet


class ConvEncoder(nn.Module):
    def __init__(self, in_dim, kernel_size, stride=1, padding=0):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_dim, in_dim // 2, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(in_dim // 2, in_dim // 4, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        return x
        # return self.conv2(F.relu(self.conv1(x)))


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param str encoder_attn_layer_type: encoder attention layer type
    :param bool macaron_style: whether to use macaron style for positionwise layer
    :param bool use_cnn_module: whether to use convolution module
    :param bool zero_triu: whether to zero the upper triangular part of attention matrix
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        encoder_attn_layer_type="mha",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        relu_type="prelu",
        a_upsample_ratio=1,
        layerscale=False,
        init_values=0.1,
        ff_bn_pre=False,
        post_norm=True,
        gamma_zero=False,
        gamma_init=0.1,
        mask_init_type=None,
        mask_std_init=0.02,
        last_linear=False,
        last_norm=True,
        odim=None,
        drop_path=0.0,
        predictor_stride=1,
        encoder_stride=1,
        multi_pred=False,
        conv_sub=False,
        mlp_sub=False,
        separate_preds=False,
        predictor_avg_pool_stride=1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if encoder_attn_layer_type == "rel_mha":
            pos_enc_class = RelPositionalEncoding
        elif encoder_attn_layer_type == "legacy_rel_mha":
            pos_enc_class = LegacyRelPositionalEncoding
        # -- frontend module.

        self.frontend_a = Conv1dResNet(
            relu_type=relu_type,
            a_upsample_ratio=a_upsample_ratio,
            gamma_zero=gamma_zero, 
            gamma_init=gamma_init,
        )

        self.frontend_v = Conv3dResNet(
            relu_type=relu_type, gamma_zero=gamma_zero, gamma_init=gamma_init
        )

        # -- backend module.
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "vanilla_linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        elif input_layer == "only_pos":
            self.embed = pos_enc_class(attention_dim, positional_dropout_rate)
        elif input_layer == "token_embed":
            self.embed = MaskEmbedding(
                idim, 
                pos_enc_class(attention_dim, positional_dropout_rate), 
                attention_dim, 
                init_type=mask_init_type, 
                std_init=mask_std_init,
                conv_sub=conv_sub,
                mlp_sub=mlp_sub,
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if encoder_attn_layer_type == "mha":
            encoder_attn_layer = MultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif encoder_attn_layer_type == "legacy_rel_mha":
            encoder_attn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                macaron_style,
                layerscale,
                init_values,
                ff_bn_pre,
                post_norm,
                drop_path,
            ),
        )
        self.after_norm = None
        if self.normalize_before and last_norm and not multi_pred:
            self.after_norm = LayerNorm(attention_dim)
        
        self.last_linear = torch.nn.Linear(attention_dim, odim) if last_linear and not multi_pred else None

        self.multi_pred = multi_pred
        self.separate_preds = separate_preds

        if multi_pred:
            self.preds = nn.ModuleList([
                nn.Sequential(LayerNorm(attention_dim), nn.Linear(attention_dim, odim)) for _ in range(num_blocks // predictor_stride)
            ])

        self.predictor_stride = predictor_stride
        self.encoder_stride = encoder_stride
        self.predictor_avg_pool_stride = predictor_avg_pool_stride

        # self.linear_av = nn.Linear(2*idim, idim)

        self.linear_a = nn.Linear(idim, attention_dim)
        self.linear_v = nn.Linear(idim, attention_dim)
        self.linear_av = nn.Linear(2*idim, attention_dim)


    def forward(self, xs_v, xs_a, masks, return_feats=False):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param str extract_features: the position for feature extraction
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        xs_v = self.frontend_v(xs_v)
        xs_a = self.frontend_a(xs_a)

        xs_av = self.linear_av(torch.cat([xs_v, xs_a], dim=-1))
        xs_v = self.linear_v(xs_v)
        xs_a = self.linear_a(xs_a)

        xs = torch.cat([xs_v, xs_a, xs_av])
        masks = torch.cat([masks, masks, masks])

        xs = self.embed(xs)

        if return_feats:
            feats = []
            for e in self.encoders:
                xs, masks = e(xs, masks)
                if isinstance(xs, tuple):
                    feat = xs[0]
                else:
                    feat = xs
                feats.append(feat[:len(xs_v)])
            feats = torch.stack(feats)
        else:
            xs, masks = self.encoders(xs, masks)
            feats = None

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.after_norm:
            xs = self.after_norm(xs)
        
        if self.last_linear:
            xs = self.last_linear(xs)

        return xs[:len(xs_v)], xs[len(xs_v):2*len(xs_v)], xs[2*len(xs_v):], masks, feats

    def forward_single(self, xs_v=None, xs_a=None, masks=None, return_feats=False):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param str extract_features: the position for feature extraction
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        assert xs_v is not None or xs_a is not None

        if xs_v is not None:
            xs_v = self.frontend_v(xs_v)
        if xs_a is not None:
            xs_a = self.frontend_a(xs_a)

        if xs_v is not None and xs_a is not None:
            xs = self.linear_av(torch.cat([xs_v, xs_a], dim=-1))
        else:
            xs = self.linear_v(xs_v) if xs_v is not None else self.linear_a(xs_a)

        xs = self.embed(xs)

        if return_feats:
            feats = []
            for e in self.encoders:
                xs, masks = e(xs, masks)
                if isinstance(xs, tuple):
                    feat = xs[0]
                else:
                    feat = xs
                feats.append(feat)
            feats = torch.stack(feats)
        else:
            xs, masks = self.encoders(xs, masks)
            feats = None

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.after_norm:
            xs = self.after_norm(xs)
        
        if self.last_linear:
            xs = self.last_linear(xs)

        return xs, masks, feats
