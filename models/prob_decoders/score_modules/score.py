
from ..base_prob_dec import BaseProbDecoder
from .score_network import ScoreNet
import torch
import numpy as np
import torch.nn as nn
from models.libs.utils import clip_norm

class ScoreMatchDecoder(BaseProbDecoder):
    def __init__(self, embed_size, layer_num, event_type_num, mean_log_inter_time: float=0.0, std_log_inter_time: float=1.0,
                 anneal_power: float=2.0, order: int=3, sigma_begin: float=5.0, sigma_end: float=0.01, noise_level_num: int=50,
                 *args, **kwargs):
        super().__init__(embed_size, layer_num, event_type_num, mean_log_inter_time, std_log_inter_time, *args, **kwargs)
        
        self.anneal_power = anneal_power
        self.order = order 

        self.score_net = ScoreNet(embed_size=embed_size, layer_num=layer_num)
        
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end),
                               noise_level_num)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

    def cumulative_risk_func(self, history_embedding, dt, sample_num=200, max_dt=5, steps=20):
        return self.empirical_cumulative_risk_func(history_embedding, dt, sample_num=sample_num, max_dt=max_dt, steps=steps)
    
    
    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape
        seq_dts = seq_dts.clamp(1e-8)
        noise_level = torch.randint(0, self.sigmas.size(0), (batch_size, seq_length, event_num)).to(seq_dts).long() # (num_graph)
        used_sigmas = self.sigmas[noise_level]

        seq_dts_expand = self.normalize_dt(seq_dts).unsqueeze(dim=-1).expand(-1, -1, event_num)
        seq_dts_expand_perturb = seq_dts_expand + torch.rand_like(seq_dts_expand)*used_sigmas
        
        target =  -1 / (used_sigmas ** 2) * (seq_dts_expand_perturb - seq_dts_expand)

        scores = self.score_net(seq_dts_expand_perturb, history_embedding)
        scores = scores * (1. / used_sigmas)

        log_loss = (0.5 * ((scores - target) ** 2) * (used_sigmas ** self.anneal_power) * seq_onehots).sum()

        mark_logits = self.compute_ce(history_embedding, seq_types)
        return log_loss, mark_logits
    
    @torch.no_grad()
    def get_score(self, seq_dts, history_embedding, sigma, *args):
        scores = self.score_net(seq_dts, history_embedding) * (1. / sigma)
        return scores
    
    @torch.no_grad()
    def Langevin_Dynamics_sampling(self, t_noise, history_embedding,
                                   n_steps_each=100, step_lr=1e-7, 
                                   clip=1.0, min_sigma=0, intermediate_output=False):
        t_vecs = []
        cnt_sigma = 0

        for i, sigma in enumerate(self.sigmas):
            if sigma < min_sigma:
                break
            cnt_sigma += 1
            step_size = step_lr * (sigma / self.sigmas[-1]) ** 2
            for step in range(n_steps_each):
                noise = torch.randn_like(t_noise) * torch.sqrt(step_size * 2)
                score_t = self.get_score(t_noise, history_embedding, sigma) 
                score_t = clip_norm(score_t, limit=clip)
                t_noise = t_noise + step_size * score_t + noise
                t_vecs.append(t_noise)
        
        if intermediate_output == True:
            t_vecs = torch.stack(t_vecs, dim=0).view(cnt_sigma, n_steps_each, -1, 1)  # (sigams, 100, -1, 1)
            # torch.save(t_vecs, f='scorematch_dynamics')
            
            return t_vecs
        
        return t_vecs[-1]

    def compute_nll(self, seq_dts, seq_onehots, history_embedding, sample_num=10, *args):
        return torch.tensor(0)
    
    def t_sample(self, history_embedding, seq_onehots=True, sample_num=100, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape
        sample_shape = torch.Size((batch_size * sample_num, seq_length, event_num, 1))
        history_embedding = history_embedding[:,None,...]\
                            .expand(-1,sample_num,-1,-1,-1)\
                            .reshape((batch_size * sample_num, seq_length, event_num, embed_size))
        t_noise = torch.randn(sample_shape[:-1]).to(history_embedding)
        sample_interval = self.Langevin_Dynamics_sampling(t_noise, history_embedding, intermediate_output=False)
        sample_interval = sample_interval.reshape(batch_size, sample_num, seq_length, event_num)
        sample_interval = self.unnormalize_dt(sample_interval)
        mask = torch.ones_like(sample_interval)

        return sample_interval, mask
    
    def inter_time_dist_pred(self, history_embedding, max_t, resolution):
        # using MC get mean, sample number = resolution
        interval, _ = self.t_sample(history_embedding, sample_num=resolution)
        interval = interval.mean(dim=1)
        return interval