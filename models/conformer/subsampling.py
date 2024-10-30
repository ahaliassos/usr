#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""

import torch

from .nets_utils import Lambda


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    :param nn.Module pos_enc_class: positional encoding layer

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc_class, use_bn=False):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        if use_bn:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2, padding=1),
                torch.nn.BatchNorm2d(odim),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2, padding=1),
                torch.nn.BatchNorm2d(odim),
                torch.nn.ReLU(),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(1, odim, 3, 2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(odim, odim, 3, 2, padding=1),
                torch.nn.ReLU(),
            )
        # if use_bn:
        #     self.out = torch.nn.Sequential(
        #         torch.nn.Linear(odim * idim // 4, odim),  # this has been changed due to padding (original didn't pad)
        #         Lambda(lambda x: x.transpose(1, 2).contiguous()),
        #         torch.nn.BatchNorm1d(odim),
        #         Lambda(lambda x: x.transpose(1, 2).contiguous()),
        #         pos_enc_class,
        #     )
        # else:
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * idim // 4, odim),  # this has been changed due to padding (original didn't pad)
            pos_enc_class,
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
               or Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        # if RelPositionalEncoding, x: Tuple[torch.Tensor, torch.Tensor]
        # else x: torch.Tensor
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        # return x, x_mask[:, :, :-2:2][:, :, :-2:2]  # this should change due to padding
        return x, x_mask  # this is if the mask is already the same as the video's
