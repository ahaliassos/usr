import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import Encoder
from .initializer import initialize
from .nets_utils import make_non_pad_mask


class Conformer(nn.Module):
    def __init__(
            self,
            in_dim,
            heads,
            depth,
            attention_dim,
            input_layer,
            attention_dropout_rate,
            transpose=False,
            use_last_linear=False,
            out_dim=256,
            dropout_rate=0.1,
            mlp_dim=2048,
            positional_dropout_rate=0.1,
            macaron_style=True,
            encoder_attn_layer_type="rel_mha",
            use_cnn_module=True,
            cnn_module_kernel=31,
            zero_triu=False,
            transformer_init="pytorch",
            rel_pos_type="latest",    # [None, "latest"]
            normalize_before=True,
            normalize_before_only_mha=False,
            normalize_res=False,
            layerscale=True,
            init_values=0.,
            post_norm=True,
            post_norm_bn=False,
            extra_bn_ff=False,
            extra_bn_conv=False,
            extra_bn_stem=False,
            ff_bn_pre=False,
            mha_bn_pre=False,
            ff_bn_res=False,
            mha_bn_res=False,
            norm_attention=False,
            norm_attention_tau=0.1,
            final_norm=True,
            pool=False,
    ):
        super(Conformer, self).__init__()

        assert rel_pos_type in [None, "latest"], "Whether to use the latest relative positional encoding or the legacy one."
        if rel_pos_type is None and encoder_attn_layer_type == "rel_mha":
            encoder_attn_layer_type = "legacy_rel_mha"
            print(
                "Using legacy_rel_pos and it will be deprecated in the future. More Details can be found in "
                "https://github.com/espnet/espnet/pull/2816."
            )

        # encoder
        encoder = Encoder(
            idim=in_dim,
            attention_dim=attention_dim,
            attention_heads=heads,
            linear_units=mlp_dim,
            num_blocks=depth,
            dropout_rate=dropout_rate,
            positional_dropout_rate=positional_dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            input_layer=input_layer,
            macaron_style=macaron_style,
            encoder_attn_layer_type=encoder_attn_layer_type,
            use_cnn_module=use_cnn_module,
            cnn_module_kernel=cnn_module_kernel,
            zero_triu=zero_triu,
            normalize_before=normalize_before,
            normalize_before_only_mha=normalize_before_only_mha,
            normalize_res=normalize_res,
            layerscale=layerscale,
            init_values=init_values,
            post_norm=post_norm,
            post_norm_bn=post_norm_bn,
            extra_bn_ff=extra_bn_ff,
            extra_bn_conv=extra_bn_conv,
            extra_bn_stem=extra_bn_stem,
            ff_bn_pre=ff_bn_pre,
            mha_bn_pre=mha_bn_pre,
            ff_bn_res=ff_bn_res,
            mha_bn_res=mha_bn_res,
            norm_attention=norm_attention,
            norm_attention_tau=norm_attention_tau,
            final_norm=final_norm,
        )
        # -- init
        initialize(encoder, init_type=transformer_init)
        self.encoder = encoder

        self.pool = pool
        self.last_linear = nn.Linear(attention_dim, out_dim) if use_last_linear else None

        self.transpose = transpose

    def forward(self, x, mask):
        if self.transpose:
            x = x.transpose(1, 2).contiguous()
        x, out_mask = self.encoder(x, mask)

        if self.pool:
            x = torch.stack([s[m].mean(dim=0) for s, m in zip(x, mask.squeeze(1))])
        x = self.last_linear(x) if self.last_linear else x

        if self.transpose and not self.pool:
            x = x.transpose(1, 2).contiguous()
        return x
