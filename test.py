from tqdm import tqdm

from datasets.tpp_loader import *
from models.prob_decoders.diffusion_modules import *
from models.prob_decoders.vae_modules import *
from models.prob_decoders.cnflow_modules import *
from models.prob_decoders.score_modules import *
from models.tpp_warper import TPPWarper
from models.embeddings import TrigonoTimeEmbedding, TypeEmbedding, PositionEmbedding
from models.hist_encoders import *
from models.prob_decoders.intensities import *
from trainers.trainer import Trainer
SEED = 2020

def SetSeed(seed):
    """function used to set a random seed
    Arguments:
        seed {int} -- seed number, will set to torch and numpy
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ =='__main__':
    SetSeed(SEED)
    device = torch.device('cuda:{}'.format(1))
    
    embed_size = 32
    layer_num = 2
    batch_size = 8
    val_batch_size = 2
    # load data
    dataset_dir = './data/stackoverflow/'
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

    hist_encoder = AttentionEnc(event_type_num=event_type_num, 
                             input_size=embed_size, 
                             embed_size=embed_size,
                             layer_num=layer_num, 
                             attention_heads=4)
                             
    # hist_encoder = LSTMEnc(event_type_num=event_type_num, 
    #                         input_size=embed_size, 
    #                         embed_size=embed_size,
    #                         layer_num=layer_num, 
    #                         attention_heads=4)

    # prob_decoder = WeibMix(embed_size=embed_size, 
    #                              layer_num=layer_num, 
    #                              event_type_num=event_type_num, 
    #                              mean_log_inter_time=mean_log_dt, 
    #                              std_log_inter_time=std_log_dt)

    # prob_decoder = LogNormMix(embed_size=embed_size, 
    #                              layer_num=layer_num, 
    #                              event_type_num=event_type_num, 
    #                              mean_log_inter_time=mean_log_dt, 
    #                              std_log_inter_time=std_log_dt)
    
    # prob_decoder = GomptMix(embed_size=embed_size, 
    #                             layer_num=layer_num, 
    #                             event_type_num=event_type_num, 
    #                             mean_log_inter_time=mean_log_dt, 
    #                             std_log_inter_time=std_log_dt)
    # prob_decoder = GaussianMix(embed_size=embed_size, 
    #                             layer_num=layer_num, 
    #                             event_type_num=event_type_num, 
    #                             mean_log_inter_time=mean_log_dt, 
    #                             std_log_inter_time=std_log_dt)
    # prob_decoder = FNNIntegral(embed_size=embed_size, 
    #                         layer_num=layer_num, 
    #                         event_type_num=event_type_num, 
    #                         mean_log_inter_time=mean_log_dt, 
    #                         std_log_inter_time=std_log_dt)
    
    # prob_decoder = VAEDecoder(embed_size=embed_size,
    #                             layer_num=layer_num, 
    #                             event_type_num=event_type_num,
    #                             mean_log_inter_time=mean_log_dt, 
    #                             std_log_inter_time=std_log_dt)
    
    # prob_decoder = CNFDecoder(embed_size=embed_size,
    #                         layer_num=layer_num, 
    #                         event_type_num=event_type_num,
    #                         mean_log_inter_time=mean_log_dt, 
    #                         std_log_inter_time=std_log_dt)
    ################ no runing


    prob_decoder = DiffusionDecoder(embed_size=embed_size,
                                layer_num=layer_num, 
                                event_type_num=event_type_num,
                                mean_log_inter_time=mean_log_dt, 
                                std_log_inter_time=std_log_dt)

    # prob_decoder = ScoreMatchDecoder(embed_size=embed_size,
    #                                 layer_num=layer_num, 
    #                                 event_type_num=event_type_num,
    #                                 mean_log_inter_time=mean_log_dt, 
    #                                 std_log_inter_time=std_log_dt)
    
    model = TPPWarper(time_embedding=time_embedding,
                     type_embedding=type_embedding,
                     position_embedding=position_embedding,
                     encoder=hist_encoder,
                     decoder=prob_decoder,
                     event_type_num=event_type_num,
                     mean_log_inter_time=mean_log_dt,
                     std_log_inter_time=std_log_dt,
                     max_dt=max_dt)
    
    # train 
    trainer = Trainer(
        data=data,
        model=model,
        seq_length=seq_lengths,
        max_t=max_t,
        max_dt=max_dt,
        log_dir='experiments/',
        experiment_name= 'att_diff_sof_2021',#.format(SEED),
        device=device,
        max_epoch=100,
        lr=1e-3
    )
    
    trainer.train()
    trainer.final_test(n=3)
    trainer.plot_similarity('type_similarity_sof')