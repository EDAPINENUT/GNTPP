from numpy import number
import torch.nn as nn
import torch
from torch.distributions import Categorical
from functools import partial

class BaseProbDecoder(nn.Module):
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
        super().__init__()
        self.layer_num = layer_num
        self.embed_size = embed_size
        self.event_type_num = event_type_num
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        mean_log_inter_time, std_log_inter_time = to_torch(mean_log_inter_time), to_torch(std_log_inter_time)
        self.register_buffer("mean_log_inter_time", mean_log_inter_time.detach().clone())
        self.register_buffer("std_log_inter_time", std_log_inter_time.detach().clone())
        
        self.mark_linear = nn.Linear(self.embed_size, self.event_type_num + 1)
    
    def t_sample(self, history_embedding, seq_onehots=None, sample_num=100, *args):
        raise NotImplementedError()
    
    def mark_logit(self, history_embedding, seq_types):
        history_embedding = history_embedding[:,:,0,...] 
        self.mark_logits = torch.log_softmax(self.mark_linear(history_embedding).squeeze(), dim=-1)  # (batch_size, seq_len, num_marks)
        mark_dist = Categorical(logits=self.mark_logits)
        mask = ~(seq_types == self.event_type_num)
        mark_dist = -(mark_dist.log_prob(seq_types) * mask).sum()
        return mark_dist
    
    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=100):
        raise NotImplementedError()
    
    def compute_ce(self, history_embedding, seq_types, *args):
        mark_logits = self.mark_logit(history_embedding, seq_types)
        return mark_logits
    
    def normalize_dt(self, seq_dts):
        seq_dts = (seq_dts + 1e-8).log() - self.mean_log_inter_time
        seq_dts = seq_dts/self.std_log_inter_time
        return seq_dts
    
    def unnormalize_dt(self, seq_dts):
        seq_dts = seq_dts * self.std_log_inter_time + self.mean_log_inter_time
        seq_dts = (seq_dts).exp() - 1e-8     
        return seq_dts.clamp(min=0.0)
    
    def empirical_cumulative_risk_func(self, history_embedding, dt, sample_num=400, max_dt=5.0, steps=40, max_t=50):
        event_type_num = history_embedding.shape[-2]
        intervals = torch.linspace(0, max_dt, steps).to(dt)
        intervals = torch.cat([intervals,torch.tensor([max_t]).to(dt)])
        dt_expand = dt[...,None].expand(-1,-1,event_type_num).unsqueeze(dim=-1)
        intervals_expand = intervals.reshape(torch.ones(len(dt_expand.shape) - 1).long().tolist() + [-1])
        gt_mask = torch.gt(dt_expand, intervals_expand)
        interval_idx = gt_mask.sum(dim=-1) - 1
        target_interval = torch.stack([intervals[interval_idx], intervals[interval_idx + 1]], dim=-1)
        samples, mask = self.t_sample(history_embedding=history_embedding, sample_num=sample_num)
        try: 
            assert mask.prod().float() == 1
            samples = samples[...,None].transpose(1,-1).squeeze(dim=1)
            gt_mask_samples = torch.gt(samples, target_interval[...,[0]])
            le_mask_samples = torch.le(samples, target_interval[...,[1]])
            total_number_in_interval = (gt_mask_samples * le_mask_samples).sum(dim=-1)
            expectation_number = total_number_in_interval/sample_num
            return expectation_number

        except:
            raise NotImplementedError('The version of mask with invalid sample is YET to be developed.')
        
    def empirical_risk_func(self, history_embedding, dt, sample_num=400, max_dt=5.0, steps=40):
        cumulative_risk = self.empirical_cumulative_risk_func(history_embedding, dt, sample_num=400, max_dt=5.0, steps=40)
        intervals = torch.linspace(0, max_dt, steps).to(dt)
        delta_t = intervals[1] - intervals[0]
        return cumulative_risk/delta_t

    def cumulative_risk_func(self, **args):
        return NotImplementedError()


        

