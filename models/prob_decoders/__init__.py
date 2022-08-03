from .ncl_modules import THPDecoder, SAHPDecoder
from .base_prob_dec import *
from .cnflow_modules import CNFDecoder
from .diffusion_modules import DiffusionDecoder
from .gan_modules import WasDiscriminator, TransGenerator
from .score_modules import ScoreMatchDecoder
from .vae_modules import VAEDecoder
from .intensities import *
from .deterministic import *
DECODER_DICT = {
    'CNF': CNFDecoder,
    'Diffusion': DiffusionDecoder,
    'GAN': TransGenerator,
    'ScoreMatch': ScoreMatchDecoder,
    'VAE': VAEDecoder,
    'LogNorm': LogNormMix,
    'Gompt': GomptMix,
    'Gaussian': GaussianMix,
    'Weibull': WeibMix,
    'FNN': FNNIntegral,
    'THP': THPDecoder, 
    'SAHP': SAHPDecoder,
    'Determ':RegHead
}

def get_decoder(**args):
    dec_name = args['prob_dec']
    return DECODER_DICT[dec_name](**args)