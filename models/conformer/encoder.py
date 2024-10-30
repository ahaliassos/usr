#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
import torch.nn as nn

from .nets_utils import rename_state_dict

from .attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from .convolution import ConvolutionModule
from .embedding import (
    PositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding, # noqa: H301
)
from .encoder_layer import EncoderLayer
from .layer_norm import LayerNorm
from .multi_layer_conv import Conv1dLinear
from .multi_layer_conv import MultiLayeredConv1d
from .nets_utils import Lambda
from .positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from .repeat import repeat
from .subsampling import Conv2dSubsampling


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
    :param int cnn_module_kernel: kernerl size of convolution module
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=12,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        normalize_before_only_mha=False,
        normalize_res=False,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        layerscale=True,
        init_values=0.,
        encoder_attn_layer_type="mha",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        post_norm=True,
        post_norm_bn=False,
        final_norm=False,
        extra_bn_ff=False,
        extra_bn_conv=False,
        extra_bn_stem=False,
        ff_bn_pre=False,
        mha_bn_pre=False,
        ff_bn_res=False,
        mha_bn_res=False,
        norm_attention=False,
        norm_attention_tau=0.1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if encoder_attn_layer_type == "rel_mha":
            pos_enc_class = RelPositionalEncoding
        elif encoder_attn_layer_type == "legacy_rel_mha":
            pos_enc_class = LegacyRelPositionalEncoding

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, dropout_rate, extra_bn_stem),
                extra_bn_stem,
            )
        elif input_layer == "vanilla_linear":
            if extra_bn_stem:
                self.embed = torch.nn.Sequential(
                    torch.nn.Linear(idim, attention_dim),
                    # Lambda(lambda x: x.transpose(1, 2).contiguous()),
                    # torch.nn.BatchNorm1d(attention_dim),
                    # Lambda(lambda x: x.transpose(1, 2).contiguous()),
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
            else:
                self.embed = torch.nn.Sequential(
                    torch.nn.Linear(idim, attention_dim),
                    pos_enc_class(attention_dim, positional_dropout_rate),
                )
        # elif input_layer == "vgg2l":
        #    self.embed = VGG2L(idim, attention_dim)
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.embed = torch.compile(self.embed)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate, extra_bn_ff)
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
                zero_triu,
            )
        elif encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer = RelPositionMultiHeadedAttention
            encoder_attn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
                norm_attention,
                norm_attention_tau,
                # extra_bn_mha,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, True, extra_bn_conv)

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                normalize_before_only_mha,
                normalize_res,
                concat_after,
                macaron_style,
                layerscale,
                init_values,
                post_norm,
                post_norm_bn,
                ff_bn_pre,
                mha_bn_pre,
                ff_bn_res,
                mha_bn_res,
            ),
        )
        self.post_norm = post_norm
        self.post_norm_bn = post_norm_bn
        self.final_norm = final_norm
        if self.normalize_before and final_norm:
            self.after_norm = nn.BatchNorm1d(attention_dim) if post_norm_bn else LayerNorm(attention_dim)

    def forward(self, xs, masks, extract_layer=0):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        if isinstance(self.embed, (Conv2dSubsampling)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if extract_layer:
            print(
                " ###Extracting from conformer layer " + str(extract_layer) + "######",
                end="\r",
            )
            layers = self.encoders[:extract_layer]
            xs, masks = layers(xs, masks)
        else:
            xs, masks = self.encoders(xs, masks)
        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before and self.final_norm:
            if self.post_norm_bn:
                xs = xs.transpose(1, 2).contiguous()
            xs = self.after_norm(xs)
            if self.post_norm_bn:
                xs = xs.transpose(1, 2).contiguous()
        return xs, masks

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        if isinstance(self.embed, Conv2dSubsampling):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)
        if cache is None:
            cache = [None for _ in range(len(self.encoders))]
        new_cache = []
        for c, e in zip(cache, self.encoders):
            xs, masks = e(xs, masks, cache=c)
            new_cache.append(xs)
        if self.normalize_before and self.final_norm:
            if self.post_norm_bn:
                xs = xs.transpose(1, 2).contiguous()
            xs = self.after_norm(xs)
            if self.post_norm_bn:
                xs = xs.transpose(1, 2).contiguous()
        return xs, masks, new_cache
