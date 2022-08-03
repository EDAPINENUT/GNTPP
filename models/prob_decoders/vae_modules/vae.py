import torch
from ..base_prob_dec import BaseProbDecoder
import torch.nn as nn
import torch.nn.functional as F

class EncoderModule(nn.Module):
    def __init__(self,
                 embed_size,
                 layer_num,
                 event_type_num,
                 mean_log_inter_time,
                 std_log_inter_time) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.event_type_num = event_type_num
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.network_t = nn.Linear(1, self.embed_size)
        self.network_h = nn.Linear(self.embed_size, self.embed_size)
        
        network = []
        for i in range(layer_num):
            network.append(nn.Linear(self.embed_size, self.embed_size))
        self.mean_network = nn.Linear(self.embed_size, self.embed_size)
        self.var_network = nn.Linear(self.embed_size, self.embed_size)     
        self.network = nn.ModuleList(network)
        self.activation = nn.GELU()
        self.training = True
        
    def mlp_transform(self, t, h):
        t_emb = self.network_t(t.unsqueeze(dim=-1))
        h_emb = self.network_h(h)
        emb = t_emb + h_emb 
        for linear in self.network:
            emb = self.activation(linear(emb))
        return emb
    
    def forward(self, t, h):
        hidden = self.mlp_transform(t, h)
        mean, log_var = self.mean_network(hidden), self.var_network(hidden)
        return mean, log_var
    
class DecoderModule(nn.Module):
    def __init__(self,
                 embed_size,
                 layer_num,
                 event_type_num,
                 mean_log_inter_time,
                 std_log_inter_time) -> None:
        super().__init__()
        self.embed_size = embed_size
        self.event_type_num = event_type_num
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.network_z = nn.Linear(self.embed_size, self.embed_size)
        self.network_h = nn.Linear(self.embed_size, self.embed_size)
        
        network = []
        for i in range(layer_num):
            network.append(nn.Linear(self.embed_size, self.embed_size))
        self.network = nn.ModuleList(network)
        self.activation = nn.GELU()
        
        self.out_network = nn.Linear(self.embed_size, 1)
    
    def mlp_transform(self, z, h):
        z_emb = self.network_z(z)
        h_emb = self.network_h(h)
        emb = z_emb + h_emb 
        for linear in self.network:
            emb = self.activation(linear(emb))
        return self.out_network(emb).squeeze(dim=-1)
        
        
    def forward(self, t, h):
        t_hat = self.mlp_transform(t, h)
        return t_hat
        
class VAEDecoder(BaseProbDecoder):
    def __init__(self,
                 embed_size,
                 layer_num,
                 event_type_num,
                 mean_log_inter_time: float=0.0,
                 std_log_inter_time: float=1.0,
                 *args,
                 **kwargs):
        super(VAEDecoder, self).__init__(
                        embed_size,
                        layer_num,
                        event_type_num,
                        mean_log_inter_time,
                        std_log_inter_time,
                        *args,
                        **kwargs)
        
        self.encoder = EncoderModule(embed_size,
                                    layer_num,
                                    event_type_num,
                                    mean_log_inter_time,
                                    std_log_inter_time)
        
        self.decoder = DecoderModule(embed_size,
                                    layer_num,
                                    event_type_num,
                                    mean_log_inter_time,
                                    std_log_inter_time)
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(mean)           
        z = mean + var*epsilon                       
        return z
        
                
    def _forward(self, t, history_embedding):
        mean, log_var = self.encoder(t, history_embedding)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        t_hat = self.decoder(z, history_embedding)
        
        return t_hat, mean, log_var
    
    
    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape
        seq_dts = seq_dts.clamp(1e-8)
        seq_dts_expand = self.normalize_dt(seq_dts).unsqueeze(dim=-1).expand(-1, -1, event_num)
        
        dts_hat, mean, log_var =  self._forward(seq_dts_expand, history_embedding)
        
        rec_loss = (seq_dts_expand - dts_hat)**2 * seq_onehots
        kl_loss = (- 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())) * seq_onehots
        log_loss = (rec_loss + kl_loss).sum()
        mark_logits = self.compute_ce(history_embedding, seq_types)
        return log_loss, mark_logits

    
    def compute_nll(self, seq_dts, seq_onehots, history_embedding, sample_num=10, *args):
        return torch.tensor(0)
    
    def t_sample(self, history_embedding, seq_onehots=None, sample_num=100, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape
        sample_shape = torch.Size((batch_size * sample_num, seq_length, event_num, 1))
        history_embedding = history_embedding[:,None,...]\
                            .expand(-1,sample_num,-1,-1,-1)\
                            .reshape((batch_size * sample_num, seq_length, event_num, embed_size))
        noise = torch.randn_like(history_embedding)
        sample_interval = self.decoder(noise, history_embedding)
        sample_interval = sample_interval.reshape(batch_size, sample_num, seq_length, event_num)
        sample_interval = self.unnormalize_dt(sample_interval)
        mask = torch.ones_like(sample_interval)

        return sample_interval, mask
    
    def cumulative_risk_func(self, history_embedding, dt, sample_num=200, max_dt=5, steps=20):
        return self.empirical_cumulative_risk_func(history_embedding, dt, sample_num=sample_num, max_dt=max_dt, steps=steps)

    def inter_time_dist_pred(self, history_embedding, max_t, resolution):
        # using MC get mean, sample number = resolution
        interval, _ = self.t_sample(history_embedding, sample_num=resolution)
        interval = interval.mean(dim=1)
        return interval