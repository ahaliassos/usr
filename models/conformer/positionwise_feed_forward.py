#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch
import torch.nn as nn


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, hidden_units, dropout_rate, use_bn=False):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.norm = nn.BatchNorm1d(hidden_units) if use_bn else None

    def forward(self, x):
        """Forward funciton."""
        if self.norm:
            x = self.w_1(x)
            x = x.transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous()
            return self.w_2(self.dropout(torch.relu(x)))
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
