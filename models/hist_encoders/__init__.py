from .attention import *
from .encoder import *
from .lstm import *

ENCODER_DICT = {
    'Attention': AttentionEnc,
    'LSTM': LSTMEnc
}

def get_encoder(**args):
    encoder_name = args['hist_enc']
    return ENCODER_DICT[encoder_name](**args)