from tqdm import tqdm

from datasets.tpp_loader import *
from models.hist_encoders.lstm import LSTMEnc
from models.tpp_warper import TPPWarper
from models.embeddings import TrigonoTimeEmbedding, TypeEmbedding, PositionEmbedding
from models.hist_encoders import LSTMEnc
from models.prob_decoders.gan_modules import *
from trainers import AdvTrainer
SEED = 2020

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

if __name__ =='__main__':
    device = torch.device('cuda:{}'.format(0))
    SetSeed(SEED)
    embed_size = 32
    layer_num = 2
    batch_size = 8
    val_batch_size = 4
    # load data
    dataset_dir = './data/synthetic_n5_c0.2/'
    data, event_type_num, seq_lengths, max_length, max_t, mean_log_dt, std_log_dt, max_dt \
        = load_dataset(dataset_dir=dataset_dir, 
                       device=device, 
                       batch_size=batch_size,
                       val_batch_size=val_batch_size)
        
    # setup model
    time_embedding = TrigonoTimeEmbedding(embed_size=embed_size//2)
    type_embedding = TypeEmbedding(event_type_num=event_type_num, 
                                   embed_size=embed_size//2, 
                                   padding_idx=event_type_num)
    
    position_embedding = PositionEmbedding(embed_size=embed_size//2,
                                           max_length=max_length)
    
    hist_encoder = LSTMEnc(event_type_num=event_type_num, 
                             input_size=embed_size,
                             embed_size=embed_size, 
                             layer_num=layer_num)
    
    prob_decoder = TransGenerator(embed_size=embed_size, 
                            layer_num=layer_num, 
                            event_type_num=event_type_num, 
                            mean_log_inter_time=mean_log_dt, 
                            std_log_inter_time=std_log_dt)
    
    model_g = TPPWarper(time_embedding=time_embedding,
                        type_embedding=type_embedding,
                        position_embedding=position_embedding,
                        encoder=hist_encoder,
                        decoder=prob_decoder,
                        event_type_num=event_type_num,
                        mean_log_inter_time=mean_log_dt,
                        std_log_inter_time=std_log_dt)
    
    model_d = WasDiscriminator(embed_size=embed_size,
                               layer_num=layer_num,
                               event_type_num=event_type_num)
    # train 
    trainer = AdvTrainer(
        data=data,
        model_g=model_g,
        model_d=model_d,
        seq_length=seq_lengths,
        max_t=max_t,
        log_dir='experiments/',
        experiment_name='att_gan_synthetic_n5_c0.2_{}'.format(SEED),
        device=device
    )
    
    trainer.train()
    trainer.final_test(n=1)