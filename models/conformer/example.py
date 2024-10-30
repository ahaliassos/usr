import torch
from .encoder import Encoder
from .nets_utils import make_non_pad_mask
from .initializer import initialize

# feature related
idim = 80

# encoder related
eunits = 2048
elayers = 6

# attention related
adim = 256
aheads = 4

# transformer specific setting
transformer_input_layer = "conv2d"
dropout_rate = 0.1
positional_dropout_rate = 0.1
attention_dropout_rate = 0.0
macaron_style = True
use_cnn_module = True
cnn_module_kernel = 31
encoder_attn_layer_type = "rel_mha"
transformer_init = "pytorch"
# -
encoder = Encoder(
    idim=idim,
    attention_dim=adim,
    attention_heads=aheads,
    linear_units=eunits,
    num_blocks=elayers,
    dropout_rate=dropout_rate,
    positional_dropout_rate=positional_dropout_rate,
    attention_dropout_rate=attention_dropout_rate,
    input_layer=transformer_input_layer,
    macaron_style=macaron_style,
    encoder_attn_layer_type=encoder_attn_layer_type,
    use_cnn_module=use_cnn_module,
    cnn_module_kernel=cnn_module_kernel,
)
# -- init
initialize(encoder, init_type=transformer_init)

B = 4
Tmax = 100
xs_pad = torch.randn(
    B, Tmax, idim
)  # torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
ilens = [
    100,
    98,
    95,
    92,
]  # torch.Tensor ilens: batch of lengths of source sequences (B)

src_mask = make_non_pad_mask(ilens).to(xs_pad.device).unsqueeze(-2)
hs_pad, hs_mask = encoder(xs_pad, src_mask)

assert hs_pad.size() == (B, ((Tmax - 1) // 2 - 1) // 2, adim)
