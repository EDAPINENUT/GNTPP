from turtle import forward
import torch
import torch.nn as nn
from ..base_prob_dec import BaseProbDecoder

class RegHead(BaseProbDecoder):
    def __init__(self,
            embed_size,
            layer_num=2,
            activation=nn.GELU(),
            event_type_num=1,
            mean_log_inter_time=0,
            std_log_inter_time=1,
            *args,
            **kwargs) -> None:
        super().__init__(embed_size, layer_num, event_type_num, mean_log_inter_time, std_log_inter_time)
        self.layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for i in range(layer_num-1)])
        self.activation = activation
        self.layer_num = layer_num
        self.layers.append(nn.Linear(embed_size, 1))

    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding, *args):
        log_loss = self.compute_loss(seq_dts, seq_onehots, history_embedding)
        mark_logits = self.compute_ce(history_embedding, seq_types)
        return log_loss, mark_logits
    
    def predict(self, history_embedding):

        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'weight'):
                layer.weight.data.clamp_(0.0)
            if hasattr(layer, 'bias'):
                layer.bias.data.clamp_(0.0)

            out = layer(history_embedding)
            out = self.activation(out) if i+1 < self.layer_num else out
        return out.squeeze(dim=-1)

    def compute_loss(self, seq_dts, seq_onehots, history_embedding, *args):
        out = self.predict(history_embedding)
        loss = (torch.square(seq_dts[...,None] - out) * seq_onehots.sum(dim=-1, keepdim=True)).sum()
        
        return loss
    
    def t_sample(self, history_embedding, seq_onehots=None, sample_num=100, *args):
        sample_interval = self.predict(history_embedding).unsqueeze(dim=1).repeat(1, sample_num, 1, 1)
        return sample_interval, torch.ones_like(sample_interval)

    def inter_time_dist_pred(self, history_embedding, max_t, resolution):
        interval = self.predict(history_embedding)
        return interval