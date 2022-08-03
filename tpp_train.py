import argparse
from pathlib import Path
import os
import json

from models.prob_decoders import *
from models.embeddings import *
from models.hist_encoders import *
from datasets.tpp_loader import *
from models.tpp_warper import TPPWarper
from trainers.trainer import Trainer
from trainers.adversarial_trainer import AdvTrainer



def SetSeed(seed):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch and numpy
    """
    import torch
    import numpy
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)

TIME_EMB = ['Trigo', 'Linear']
PROB_DEC = ['CNF','Diffusion','GAN','ScoreMatch','VAE','LogNorm','Gompt','Gaussian','Weibull','FNN', 'THP', 'SAHP']

# NOTE: The given THP and SAHP use different type-modeling methods (type-wise intensity modelling), while others model all the type in a single sequence.
# So the final metric evaluation will be in a different protocol.

HIST_ENC = ['LSTM', 'Attention']

parser = argparse.ArgumentParser(prog="Attentive Diffusion Temporal Point Process (training)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Basic
parser.add_argument('--log_dir', type=str, metavar='DIR', 
                    help='Directory where models and logs will be saved.', default='experiments/')
parser.add_argument('--dataset_dir', type=str, metavar='DIR', default='./data/stackoverflow/',
                    choices=['./data/mooc/', './data/retweet/', './data/stackoverflow/', './data/synthetic_n5_c0.2/', './data/yelp/'],
                    help='Directory for dataset.')

# Training
parser.add_argument('--max_epoch', type=int, metavar='NUM', default=100,
                    help='The maximum epoch number for training.')
parser.add_argument('--lr', type=int, metavar='RATE', default=1e-3,
                    help='The leanring rate for training.')
parser.add_argument('--load_epoch', type=int, metavar='NUM', default=0,
                    help='Load the saved epoch number for continously training.')  

parser.add_argument('--batch_size', type=int, metavar='SIZE', default=16,
                    help='Batch size for training.')        
parser.add_argument('--val_batch_size', type=int, metavar='SIZE', default=8,
                    help='Batch size for validation, which should be smaller than training batch size because some metric requires MCMC sampling.')
parser.add_argument('--experiment_name', type=str, metavar='SIZE', default=None,
                    help='The experiment name, where the file where logs and models are saved will be called.') 

# Model
parser.add_argument('--time_emb', type=str, metavar='NAME', default='Trigo', choices=TIME_EMB,
                    help='The time embedding which is used, choosen from {}.'.format(TIME_EMB))
parser.add_argument('--hist_enc', type=str, metavar='NAME', default='Attention', choices=HIST_ENC,
                    help='The history encoder which is used, choosen from {}.'.format(HIST_ENC))
parser.add_argument('--prob_dec', type=str, metavar='NAME', default='THP', choices=PROB_DEC,
                    help='The probabilistic decoder which is used, choosen from {}.'.format(PROB_DEC))
parser.add_argument('--embed_size', type=int, metavar='SIZE', default=32,
                    help='Hidden dimension for the model.')
parser.add_argument('--layer_num', type=int, metavar='NUM', default=1,
                    help='Layer number for the model.')  
parser.add_argument('--attention_heads', type=int, metavar='SIZE', default=4,
                    help='Attention heads for the attention history encoder, which should be set as a divisor of embed size.')          

###
parser.add_argument('--gpu', type=int, metavar='DEVICE', default=6,
                    help='Gpu to use for training.')
parser.add_argument('--seed', type=int, metavar='SEED', default=42,
                    help='Random seed for training.')

args = parser.parse_args()
args = vars(args)

if __name__ == '__main__':
    SetSeed(args['seed'])

    device = torch.device('cuda:{}'.format(args['gpu'])) if torch.cuda.is_available() else 'cpu'

    data, event_type_num, seq_lengths, max_length, max_t, mean_log_dt, std_log_dt, max_dt \
        = load_dataset(**args, device=device)

    args['event_type_num'] = int(event_type_num)
    args['max_length'] = int(max_length)
    args['max_t'] = max_t
    args['mean_log_dt'] = mean_log_dt
    args['std_log_dt'] = std_log_dt
    args['max_dt'] = max_dt
    
    if args['experiment_name'] == None:
        args['experiment_name'] = '{}_{}_{}_{}'.format(args['hist_enc'],
                                                       args['prob_dec'],
                                                       args['dataset_dir'].split('/')[-2],
                                                       args['seed'])

    path = Path(args['log_dir'])/args['experiment_name']
    path.mkdir(exist_ok = True, parents = True)
    sv_param = os.path.join(path, 'model_param.json')
    with open(sv_param, 'w') as file_obj:
        json.dump(args, file_obj)
    

    time_embedding, type_embedding, position_embedding = get_embedding(**args)
    hist_encoder = get_encoder(**args)
    prob_decoder = get_decoder(**args)

    model = TPPWarper(time_embedding=time_embedding,
                    type_embedding=type_embedding,
                    position_embedding=position_embedding,
                    encoder=hist_encoder,
                    decoder=prob_decoder,
                    **args)

    trainer = Trainer(
        data=data,
        model=model,
        seq_length=seq_lengths,
        device=device,
        **args
    )

    if args['prob_dec'] == 'GAN':
        model_d = WasDiscriminator(**args)
        trainer = AdvTrainer(
        data=data,
        model_g=model,
        model_d=model_d,
        seq_length=seq_lengths,
        device=device,
        **args
    )

    trainer.train()
    trainer.final_test(n=1)
    trainer.plot_similarity('type_similarity_sof')