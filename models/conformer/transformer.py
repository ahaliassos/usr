from .encoder import Encoder
from .initializer import initialize


def get_transformer():
    # feature related
    idim = 512

    # encoder related
    eunits = 2048
    elayers = 6

    # attention related
    adim = 512
    aheads = 4

    # transformer specific setting
    transformer_input_layer = "linear"
    dropout_rate = 0.1
    positional_dropout_rate = 0.1
    attention_dropout_rate = 0.0
    macaron_style = False
    use_cnn_module = False
    cnn_module_kernel = 31
    encoder_attn_layer_type = "mha"
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
    return encoder
