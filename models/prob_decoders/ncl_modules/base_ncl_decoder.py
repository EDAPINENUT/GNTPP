from numpy import number
import torch.nn as nn
from .integral import *
from torch.distributions import Categorical
from functools import partial
from ..base_prob_dec import BaseProbDecoder

class BaseNCLDecoer(BaseProbDecoder):
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
    
    def t_sample(self, history_embedding, seq_onehots=None, sample_num=100, *args):
        return 0

    def cumulative_risk_func(self, history_embedding, seq_dts, steps, max_dt, *args):
        temp_hid = self.his_to_param(history_embedding)
        batch_size, seq_len, param_num, event_type_num = temp_hid.size()

        time_step = 1 / steps
        x_axis = torch.linspace(0, max_dt, steps).to(temp_hid)
        taus = x_axis[None,None,None,:].expand(batch_size, seq_len, event_type_num, -1).detach()
        seq_dts = seq_dts[...,None, None]

        diff_dts = seq_dts * taus

        intensity = self.intensity_func(diff_dts, temp_hid.unsqueeze(dim=-1))
        integral = torch.sum(time_step * intensity, dim=-1)
        return integral

    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=500):
        temp_hid = self.his_to_param(history_embedding)
        batch_size, seq_len, param_num, event_type_num = temp_hid.size()

        return inter_time_dist_pred(self.intensity_func, temp_hid, max_t, resolution)
    
    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding):
        log_loss = self.compute_nll(seq_dts, seq_onehots, history_embedding)
        mark_logits = self.compute_ce(history_embedding, seq_types)
        return log_loss, mark_logits


        

