from .denoise_network import *
from .gaussian_diffusion import *
from ..base_prob_dec import BaseProbDecoder

class DiffusionDecoder(BaseProbDecoder):
    def __init__(self, 
                embed_size, 
                layer_num, 
                event_type_num, 
                mean_log_inter_time: float=0.0, 
                std_log_inter_time: float=1.0, 
                *args,
                **kwargs):
        super().__init__(embed_size, layer_num, event_type_num, mean_log_inter_time, std_log_inter_time, *args, **kwargs)
        
        self.denoise_fn = DenoiseNet(embed_size=embed_size, layer_num=layer_num, *args, **kwargs)
        
        self.diffusion_net = GaussianDiffusion(denoise_fn=self.denoise_fn, *args, **kwargs)
    
    def cumulative_risk_func(self, history_embedding, dt, sample_num=200, max_dt=5, steps=20):
        return self.empirical_cumulative_risk_func(history_embedding, dt, sample_num=sample_num, max_dt=max_dt, steps=steps)
    
    
    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape
        seq_dts = seq_dts.clamp(1e-8)
        seq_dts_expand = self.normalize_dt(seq_dts).unsqueeze(dim=-1).expand(-1, -1, event_num)
        
        if event_num == 1:
            seq_onehots = seq_onehots.sum(dim=-1, keepdim=True)
            
        log_loss = self.diffusion_net.log_prob(seq_dts_expand, cond=history_embedding, mask=seq_onehots)
        mark_logits = self.compute_ce(history_embedding, seq_types)
        return log_loss, mark_logits
    
    
    def t_sample(self, history_embedding, seq_onehots=True, sample_num=100, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape
        sample_shape = torch.Size((batch_size * sample_num, seq_length, event_num, 1))
        history_embedding = history_embedding[:,None,...]\
                            .expand(-1,sample_num,-1,-1,-1)\
                            .reshape((batch_size * sample_num, seq_length, event_num, embed_size))
        sample_interval = self.diffusion_net.sample(cond=history_embedding, sample_shape=sample_shape)
        sample_interval = sample_interval.reshape(batch_size, sample_num, seq_length, event_num)
        sample_interval = self.unnormalize_dt(sample_interval)
        mask = torch.ones_like(sample_interval)

        return sample_interval, mask
    
    def inter_time_dist_pred(self, history_embedding, max_t, resolution):
        # using MC get mean, sample number = resolution
        interval, _ = self.t_sample(history_embedding, sample_num=resolution)
        interval = interval.mean(dim=1)
        return interval

    def compute_nll(self, seq_dts, seq_onehots, history_embedding, sample_num=10, *args):
    #     if len(seq_dts.shape) < 3:
    #         seq_dts = seq_dts.reshape(*seq_dts.shape, *(1,) * (3-len(seq_dts.shape)))
        
    #     seq_dts = self.normalize_dt(seq_dts)
        
    #     batch_size, seq_length, event_num, embed_size = history_embedding.shape

    #     history_embedding = history_embedding[:,None,...]\
    #                         .expand(-1,sample_num,-1,-1,-1)\
    #                         .reshape((batch_size * sample_num, seq_length, event_num, embed_size))

    #     shape = (batch_size * sample_num, seq_length, event_num, 1)
        
    #     mean, std = self.diffusion_net.conditional_nll_param(shape, history_embedding)
    #     mean = mean.reshape(batch_size, sample_num, seq_length, event_num).mean(dim=1)

    #     seq_dts = seq_dts.expand_as(mean)
    
    #     gaussian_nll = (std * math.sqrt(2 * math.pi)).log() + (seq_dts - mean)**2/(2*std**2)
    #     transfer_nll = self.std_log_inter_time * seq_dts + self.mean_log_inter_time + self.std_log_inter_time.log()
    #     mask_nll =  (gaussian_nll + transfer_nll) * seq_onehots
    #     return mask_nll.sum()
        return torch.tensor(0)