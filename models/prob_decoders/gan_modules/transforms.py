import torch
from torch.distributions import beta
import torch.nn as nn
from torch.nn.modules.activation import LeakyReLU
from models.basic_layers.normlayer import LayerNorm
import torch.nn.functional as F
from ..base_prob_dec import BaseProbDecoder

class TransGenerator(BaseProbDecoder):
    def __init__(self, 
                 embed_size,
                 layer_num,
                 event_type_num,
                 mean_log_inter_time: float=0.0,
                 std_log_inter_time: float=1.0,
                 *args,
                 **kwargs) -> None:
        super().__init__(embed_size,
                        layer_num,
                        event_type_num,
                        mean_log_inter_time,
                        std_log_inter_time,
                        *args,
                        **kwargs)
        
        self.network_t = nn.Linear(1, self.embed_size)
        self.network_h = nn.Linear(self.embed_size, self.embed_size)
        
        network = []
        for i in range(layer_num):
            network.append(nn.Linear(self.embed_size, self.embed_size))
        self.out_network = nn.Linear(self.embed_size, 1)            
        self.network = nn.ModuleList(network)
        self.activation = nn.GELU()    
    
    def mlp_transform(self, basic_t, h):
        t_emb = self.network_t(basic_t.unsqueeze(dim=-1))
        h_emb = self.network_h(h)
        emb = t_emb + h_emb 
        for linear in self.network:
            emb = self.activation(linear(emb))
        return F.softplus(self.out_network(emb)).squeeze(dim=-1)
    
    def generate_basic_interval(self, size:torch.Size):
        return -(torch.rand(size=size)+ 1e-8).log()
    
    def cumulative_risk_func(self, history_embedding, dt, sample_num=200, max_dt=5, steps=20):
        return self.empirical_cumulative_risk_func(history_embedding, dt, sample_num=sample_num, max_dt=max_dt, steps=steps)

    def t_sample(self, history_embedding, seq_onehots=None, sample_num=1, max_t=50):
        size = (sample_num, ) + history_embedding.shape[:-1]
        history_embedding = history_embedding.unsqueeze(dim=0)
        basic_t = self.generate_basic_interval(size=size).to(history_embedding)
        samples = self.mlp_transform(basic_t, history_embedding).transpose(0,1)
        
        if sample_num == 1:
            samples = samples.squeeze(dim=1)
            return samples, torch.ones_like(samples)
        
        return samples, torch.ones_like(samples)
    
    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=100):
        sample_t, _ = self.t_sample(history_embedding, seq_onehots=None, sample_num=resolution, max_t=max_t)
        return sample_t.mean(dim=1)
        
        
        
    
     
        