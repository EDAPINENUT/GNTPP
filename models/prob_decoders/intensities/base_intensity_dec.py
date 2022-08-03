from numpy import number
import torch.nn as nn
import torch
from torch.distributions import Categorical
from functools import partial
from ..base_prob_dec import BaseProbDecoder

class BaseIntensityDecoder(BaseProbDecoder):
    def __init__(
            self,
            embed_size,
            layer_num,
            event_type_num,
            mean_log_inter_time,
            std_log_inter_time,
            *args,
            **kwargs
        ):
        super().__init__(embed_size, layer_num, event_type_num, mean_log_inter_time, std_log_inter_time, *args, **kwargs)

    
    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding, *args):
        log_loss = self.compute_nll(seq_dts, seq_onehots, history_embedding)
        mark_logits = self.compute_ce(history_embedding, seq_types)
        return log_loss, mark_logits
    
    def compute_nll(self, seq_dts, seq_onehots, history_embedding, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape

        inter_time_dist = self.get_inter_time_dist(history_embedding)
        seq_dts = seq_dts.clamp(1e-8)
        seq_dts_expand = seq_dts[:,:,None].expand(batch_size, seq_length, event_num)

        log_intensity = inter_time_dist.log_intensity(seq_dts_expand)
        one_hot_mask = seq_onehots
        log_intensity = log_intensity * one_hot_mask
        int_intensity = inter_time_dist.int_intensity(seq_dts_expand)
        log_loss = (-log_intensity + int_intensity).sum(dim=-1) * seq_onehots.sum(dim=-1)
        return log_loss.sum()
    
    def inter_time_dist_pred(self, history_embedding, *args):
        inter_time_dist = self.get_inter_time_dist(history_embedding)
        return inter_time_dist.mean()
    
    def t_sample(self, history_embedding, seq_onehots=True, sample_num=100, max_t=50, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape

        inter_time_dist = self.get_inter_time_dist(history_embedding)
        
        return inter_time_dist.sample(sample_num=sample_num)

    def cumulative_risk_func(self, history_embedding, seq_dts, *args):
        batch_size, seq_len, event_type_num, embed_size = history_embedding.shape
        inter_time_dist = self.get_inter_time_dist(history_embedding)
        seq_dts_expand = seq_dts.unsqueeze(dim=-1).expand(batch_size, seq_len, event_type_num)

        return inter_time_dist.int_intensity(seq_dts_expand)


        

