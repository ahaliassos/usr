#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import copy
import torch

from torch import nn

from .layer_norm import LayerNorm


class EncoderLayer(nn.Module):
    """Encoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.
        MultiHeadedAttention self_attn: self attention module
        RelPositionMultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward:
        feed forward module
    :param espnet.nets.pytorch_backend.transformer.convolution.
        ConvolutionModule feed_foreard:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_when: when to normalize
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param bool macaron_style: whether to use macaron style for PositionwiseFeedForward

    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        conv_module,
        dropout_rate,
        normalize_before=True,
        normalize_before_only_mha=False,
        normalize_res=False,
        concat_after=False,
        macaron_style=False,
        layerscale=True,
        init_values=0.,
        post_norm=True,
        post_norm_bn=False,
        ff_bn_pre=False,
        mha_bn_pre=False,
        ff_bn_res=False,
        mha_bn_res=False,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.layerscale = layerscale
        self.ff_scale = 1.0
        self.conv_module = conv_module
        self.macaron_style = macaron_style
        self.normalize_before = normalize_before
        self.normalize_before_only_mha = normalize_before_only_mha
        if normalize_before:
            if not normalize_before_only_mha:
                self.norm_ff = nn.BatchNorm1d(size) if ff_bn_pre else LayerNorm(size)  # for the FNN module
            self.norm_mha = nn.BatchNorm1d(size) if mha_bn_pre else LayerNorm(size)  # for the MHA module
        if layerscale:
            self.gamma_ff = nn.Parameter(init_values * torch.ones((size,)), requires_grad=True)
            self.gamma_mha = nn.Parameter(init_values * torch.ones((size,)), requires_grad=True)
        if self.macaron_style:
            self.feed_forward_macaron = copy.deepcopy(feed_forward)
            self.ff_scale = 0.5
            # for another FNN module in macaron style
            if normalize_before and not normalize_before_only_mha:
                self.norm_ff_macaron = nn.BatchNorm1d(size) if ff_bn_pre else LayerNorm(size)
            if layerscale:
                self.gamma_ff_macaron = nn.Parameter(init_values * torch.ones((size,)), requires_grad=True)
        self.post_norm = post_norm
        if self.conv_module is not None:
            if normalize_before and not normalize_before_only_mha:
                self.norm_conv = nn.BatchNorm1d(size) if ff_bn_pre else LayerNorm(size)  # for the CNN module
            if layerscale:
                self.gamma_conv = nn.Parameter(init_values * torch.ones((size,)), requires_grad=True)
            if post_norm:
                self.norm_final = nn.BatchNorm1d(size) if post_norm_bn else LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.post_norm_bn = post_norm_bn
        self.ff_bn_pre = ff_bn_pre
        self.mha_bn_pre = mha_bn_pre
        self.ff_bn_res = ff_bn_res
        self.mha_bn_res = mha_bn_res

        self.normalize_res = normalize_res
        if normalize_res:
            if ff_bn_res:
                self.norm_ff_macaron_res = nn.BatchNorm1d(size)
                self.norm_conv_res = nn.BatchNorm1d(size)
                self.norm_ff_res = nn.BatchNorm1d(size)
            else:
                self.norm_ff_macaron_res = LayerNorm(size)
                self.norm_conv_res = LayerNorm(size)
                self.norm_ff_res = LayerNorm(size)

            self.norm_mha_res = nn.BatchNorm1d(size) if mha_bn_res else LayerNorm(size)

    def forward(self, x_input, mask, cache=None):
        """Compute encoded features.

        :param torch.Tensor x_input: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :param torch.Tensor cache: cache for x (batch, max_time_in - 1, size)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None
        # whether to use macaron style
        if self.macaron_style:
            residual = x
            if self.normalize_before and not self.normalize_before_only_mha:
                if self.ff_bn_pre:
                    x = x.transpose(1, 2).contiguous()
                x = self.norm_ff_macaron(x)
                if self.ff_bn_pre:
                    x = x.transpose(1, 2).contiguous()

            x = self.feed_forward_macaron(x)

            if self.normalize_res:
                if self.ff_bn_res:
                    x = x.transpose(1, 2).contiguous()
                x = self.norm_ff_macaron_res(x)
                if self.ff_bn_res:
                    x = x.transpose(1, 2).contiguous()

            if self.layerscale:
                x = residual + self.ff_scale * self.dropout(self.gamma_ff_macaron * x)
            else:
                x = residual + self.ff_scale * self.dropout(x)

            # if self.normalize_when == "after":
            #     if self.ff_bn:
            #         x = x.transpose(1, 2).contiguous()
            #     x = self.norm_ff_macaron(x)
            #     if self.ff_bn:
            #         x = x.transpose(1, 2).contiguous()

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            if self.mha_bn_pre:
                x = x.transpose(1, 2).contiguous()
            x = self.norm_mha(x)
            if self.mha_bn_pre:
                x = x.transpose(1, 2).contiguous()

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.normalize_res:
            if self.mha_bn_res:
                x_att = x_att.transpose(1, 2).contiguous()
            x_att = self.norm_mha_res(x_att)
            if self.mha_bn_res:
                x_att = x_att.transpose(1, 2).contiguous()

        if self.concat_after:
            assert False
            x_concat = torch.cat((x, x_att), dim=-1)
            if self.layerscale:
                x = residual + self.gamma_mha * self.concat_linear(x_concat)
            else:
                x = residual + self.concat_linear(x_concat)
        else:
            if self.layerscale:
                x = residual + self.dropout(self.gamma_mha * x_att)
            else:
                x = residual + self.dropout(x_att)
        # if self.normalize_when == "after":
        #     if self.mha_bn:
        #         x = x.transpose(1, 2).contiguous()
        #     x = self.norm_mha(x)
        #     if self.mha_bn:
        #         x = x.transpose(1, 2).contiguous()

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before and not self.normalize_before_only_mha:
                if self.ff_bn_pre:
                    x = x.transpose(1, 2).contiguous()
                x = self.norm_conv(x)
                if self.ff_bn_pre:
                    x = x.transpose(1, 2).contiguous()

            x = self.conv_module(x)

            if self.normalize_res:
                if self.ff_bn_res:
                    x = x.transpose(1, 2).contiguous()
                x = self.norm_conv_res(x)
                if self.ff_bn_res:
                    x = x.transpose(1, 2).contiguous()

            if self.layerscale:
                x = residual + self.dropout(self.gamma_conv * x)
            else:
                x = residual + self.dropout(x)

            # if self.normalize_when == "after":
            #     if self.ff_bn:
            #         x = x.transpose(1, 2).contiguous()
            #     x = self.norm_conv(x)
            #     if self.ff_bn:
            #         x = x.transpose(1, 2).contiguous()

        # feed forward module
        residual = x
        if self.normalize_before and not self.normalize_before_only_mha:
            if self.ff_bn_pre:
                x = x.transpose(1, 2).contiguous()
            x = self.norm_ff(x)
            if self.ff_bn_pre:
                x = x.transpose(1, 2).contiguous()

        x = self.feed_forward(x)

        if self.normalize_res:
            if self.ff_bn_res:
                x = x.transpose(1, 2).contiguous()
            x = self.norm_ff_res(x)
            if self.ff_bn_res:
                x = x.transpose(1, 2).contiguous()

        if self.layerscale:
            x = residual + self.ff_scale * self.dropout(self.gamma_ff * x)
        else:
            x = residual + self.ff_scale * self.dropout(x)

        # if self.normalize_when == "after":
        #     if self.ff_bn:
        #         x = x.transpose(1, 2).contiguous()
        #     x = self.norm_ff(x)
        #     if self.ff_bn:
        #         x = x.transpose(1, 2).contiguous()

        if self.conv_module is not None and self.post_norm:
            if self.post_norm_bn:
                x = x.transpose(1, 2).contiguous()
            x = self.norm_final(x)
            if self.post_norm_bn:
                x = x.transpose(1, 2).contiguous()

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask
        else:
            return x, mask
