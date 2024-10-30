#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Parameter initialization."""

import torch
import logging

from .layer_norm import LayerNorm


def initialize(model, init_type="pytorch"):
    """Initialize Transformer module.

    :param torch.nn.Module model: transformer instance
    :param str init_type: initialization type
    """
    if init_type == "pytorch":
        return

    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init_type == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init_type == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init_type == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init_type == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init_type)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()

    # reset some modules with default init
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, LayerNorm)):
            m.reset_parameters()


def initialize_encoder(model, transformer_input_layer="conv2d"):

    if transformer_input_layer == "conv2d":
        return

    assert transformer_input_layer in [
        "conv1d",
        "conv3d",
    ], "Only support encoder initialisation from raw input."

    model_dict = model.encoder.state_dict()
    logging.info(
        "Loading pretrained student encoder (pretrained using LRW dataset), [BEFORE] norm2weight: {}".format(
            calculateNorm2(model.encoder)
        )
    )
    model_dict = model.encoder.state_dict()
    pretrain_path = get_lrw_path_config(input_layer=transformer_input_layer)
    pretrained_dict = torch.load(pretrain_path)["model_state_dict"]
    matching_dict = {}
    matched = []
    mismatched = []
    for k, v in pretrained_dict.items():
        if "embed.trunk." + k in model_dict:
            matching_dict["embed.trunk." + k] = v
            matched.append(k)
        else:
            mismatched.append(k)
    logging.info("matched key ( seletected 20 keys): {}".format(matched[:20]))
    logging.info("mismatched key (selected 20 keys): {}".format(mismatched[:20]))
    model_dict.update(matching_dict)
    model.encoder.load_state_dict(model_dict)
    logging.info(
        "Loading pretrained student encoder (pretrained using LRW dataset), [AFTER] norm2weight: {}".format(
            calculateNorm2(model.encoder)
        )
    )
