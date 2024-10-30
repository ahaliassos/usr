#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

from einops import rearrange
import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict, pad_list
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
from espnet.nets.pytorch_backend.transformer.mobile_encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
# from espnet.nets.pytorch_backend.transformer.positional_bias import AlibiPositionalBias, LearnedAlibiPositionalBias
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.raw_embeddings import VideoEmbedding
from espnet.nets.pytorch_backend.transformer.raw_embeddings import AudioEmbedding
from espnet.nets.pytorch_backend.backbones.conv3d_extractor  import Conv3dResNet
from espnet.nets.pytorch_backend.backbones.conv1d_extractor  import Conv1dResNet


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


# def get_bias(i, j, device):
#     i_arange = torch.arange(j - i, j, device = device)
#     j_arange = torch.arange(j, device = device)
#     bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
#     return bias


def get_bias(x):
    return -torch.abs(rearrange(x, 'b j -> b 1 1 j') - rearrange(x, 'b i -> b 1 i 1'))


class CustomLinear(torch.nn.Linear):
    def forward(self, x, pos):
        return super().forward(x), pos


class MySequential(torch.nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class MobileEncoder(torch.nn.Module):
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
        odim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer=None,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        encoder_attn_layer_type="mha",
        zero_triu=False,
        padding_idx=-1,
        relu_type="prelu",
        a_upsample_ratio=1,
        layerscale=False,
        init_values=0.,
        ff_bn_pre=False,
        post_norm=False,
        drop_path=0.,
        mask_init_type=None,
        mask_std_init=0.02,
        final_norm=True,
        remove_masked_tokens=False,
        use_tiny=False,
        predictor_stride=1,
    ):
        """Construct an Encoder object."""
        super(MobileEncoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        if encoder_attn_layer_type == "rel_mha":
            pos_enc_class = RelPositionalEncoding
        elif encoder_attn_layer_type == "legacy_rel_mha":
            pos_enc_class = LegacyRelPositionalEncoding

        self.remove_masked_tokens = remove_masked_tokens
        self.encoder_attn_layer_type = encoder_attn_layer_type

        self.idim = idim
        self.size = attention_dim

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        else:
            raise NotImplementedError("Support only linear.")

        # self.use_alibi = use_alibi
        # alibi = None
        # if use_alibi:
        #     alibi = LearnedAlibiPositionalBias(attention_heads) if learned_alibi else AlibiPositionalBias(attention_heads)
        
        self.embed = None
        if input_layer == "token_embed":
            self.embed = MaskEmbedding(
                idim, 
                pos_enc_class(attention_dim if use_tiny else idim, positional_dropout_rate),
                init_type=mask_init_type, 
                std_init=mask_std_init,
            )

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
                None if use_tiny else idim,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + encoder_attn_layer)

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                idim,
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                nn.ModuleList([positionwise_layer(*positionwise_layer_args) for _ in range(2 if use_tiny else 4)]),
                dropout_rate,
                normalize_before,
                concat_after,
                layerscale,
                init_values,
                ff_bn_pre,
                post_norm,
                drop_path,
                use_tiny,
            ),
        )

        self.preds = nn.ModuleList([
            nn.Sequential(LayerNorm(idim), nn.Linear(idim, odim)) for _ in range(num_blocks // predictor_stride)
        ])

        self.predictor_stride = predictor_stride

        
    def forward(self, xs, masks, token_mask=None):
        """Encode input sequence.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param str extract_features: the position for feature extraction
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """

        if isinstance(self.embed, MaskEmbedding):
            xs = self.embed(xs, token_mask)
        else:
            xs = self.embed(xs) if self.embed else xs

        # xs_size = xs[0].size(1) if isinstance(xs, tuple) else xs.size(1)
        # xs_device = xs[0].device if isinstance(xs, tuple) else xs.device
        # mask_indices = torch.arange(xs_size, device=xs_device).unsqueeze(0)
        
        # alibi_bias = get_bias(mask_indices) if self.use_alibi else None

        # out = (xs, masks, alibi_bias)
        out = (xs, masks)

        feats = []
        for i, e in enumerate(self.encoders, 1):
            out = e(*out)
            feat = out[0]
            if isinstance(feat, tuple):
                feat = feat[0]
            if i % self.predictor_stride == 0:
                feats.append(self.preds[i // self.predictor_stride - 1](feat))
        feats = torch.stack(feats)

        # feats = []
        # for e, p in zip(self.encoders, self.preds):
        #     out = e(*out)
        #     feat = out[0]
        #     if isinstance(feat, tuple):
        #         feat = feat[0]
        #     feats.append(p(feat))
        # feats = torch.stack(feats)

        return feats, None

    def forward_one_step(self, xs, masks, cache=None):
        """Encode input frame.

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :param List[torch.Tensor] cache: cache tensors
        :return: position embedded tensor, mask and new cache
        :rtype Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        if isinstance(self.frontend, (Conv1dResNet, Conv3dResNet)):
            xs = self.frontend(xs)

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
        if self.after_norm:
            xs = self.after_norm(xs)
        return xs, masks, new_cache
