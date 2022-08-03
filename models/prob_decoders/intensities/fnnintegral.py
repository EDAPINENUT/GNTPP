import torch.nn as nn
import torch
import numpy as np
from .base_intensity_dec import BaseIntensityDecoder
import torch.nn.functional as F
from functools import partial
from models.libs.sampling import rejection_sampling

class NonnegativeLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        # Make weight non-negative at initialization
        self.weight.data.abs_()
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.weight.data.clamp_(0.0)
        self.bias.data.clamp_(0.0)
        return F.linear(input, self.weight, self.bias)

class FNNIntegral(BaseIntensityDecoder):
    def __init__(
            self,
            embed_size,
            layer_num,
            event_type_num,
            mean_log_inter_time: float=0.0,
            std_log_inter_time: float=1.0,
            *args,
            **kwargs
        ):
        super().__init__(embed_size, layer_num, event_type_num, mean_log_inter_time, std_log_inter_time, *args, **kwargs)

        self.hidden_embed_size = embed_size
        self.linear_layers = nn.ModuleList([
            NonnegativeLinear(embed_size, embed_size) for _ in range(self.layer_num - 1)
        ])
        self.linear_time = NonnegativeLinear(1, embed_size)
        self.final_layer = NonnegativeLinear(embed_size, 1)
        self.linear_rnn = nn.Linear(embed_size, embed_size, bias=False)

        # self.diag_fn = Diagonal(event_type_num + 1, embed_size)
        self.register_buffer('base_int', nn.Parameter(torch.rand(1)[0]))
  
    def mlp(self, tau, history_embedding=None):
        tau = tau.unsqueeze(-1)
        hidden = self.linear_time(tau)
        if history_embedding is not None:
            hidden += self.linear_rnn(history_embedding)
        hidden = torch.tanh(hidden)

        for linear in self.linear_layers:
            hidden = torch.tanh(linear(hidden))
        hidden = self.final_layer(hidden) + self.base_int.abs() * tau
        return F.softplus(hidden.squeeze(-1))

    def cdf(self, tau, h=None):
        integral = self.mlp(tau, h)
        return -torch.expm1(-integral)

    def pdf(self, tau, h=None):
        with torch.set_grad_enabled(True):
            tau.requires_grad_()
            integral = self.mlp(tau, h)
            intensity = torch.autograd.grad(integral, tau, torch.ones_like(integral), create_graph=True)[0]
            proba = intensity * (-integral).exp()
            return proba
        
    
    def log_cdf(self, tau, h=None):
        return torch.log(self.cdf(tau, h) + 1e-8)

    def cumulative_risk_func(self, history_embedding, seq_dts, *args):
        batch_size, seq_len, event_type_num, embed_size = history_embedding.shape
        tau = seq_dts.unsqueeze(dim=-1).detach().expand(batch_size, seq_len, event_type_num)
        integral = self.mlp(tau, history_embedding)
        return integral

    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding):
        log_loss = self.compute_nll(seq_dts, seq_onehots, history_embedding)
        mark_logits = self.compute_ce(history_embedding, seq_types)
        return log_loss, mark_logits
    
    def compute_nll(self, seq_dts, seq_onehots, history_embedding, *args):
        
        batch_size, seq_len, event_type_num, embed_size = history_embedding.shape
        seq_dts = seq_dts.clamp(1e-8)

        tau = seq_dts.unsqueeze(dim=-1).detach().expand(batch_size, seq_len, event_type_num)

        tau.requires_grad_()
        with torch.set_grad_enabled(True):
            integral = self.mlp(tau, history_embedding)
            intensity = torch.autograd.grad(integral, tau, torch.ones_like(integral), create_graph=True)[0]
            log_intensity = (intensity).log() * seq_onehots
            log_loss =  -log_intensity + integral
            log_loss = (log_loss * seq_onehots).sum(dim=-1) * seq_onehots.sum(dim=-1)
            
        return log_loss.sum()
        

    def inter_time_dist_pred(self, history_embedding, max_t=40, resolution=500):
        with torch.set_grad_enabled(True):
            batch_size, seq_len, event_type_num, embed_size = history_embedding.shape
            time_step = max_t / resolution
            x_axis = torch.linspace(0, max_t, resolution).to(history_embedding)
            taus = x_axis[None,None,None,:].expand(batch_size, seq_len, event_type_num, -1).detach()
            taus.requires_grad_() 
            integral = self.mlp(taus, history_embedding[:,:,:,None,:])

            intensity = torch.autograd.grad(integral, taus, torch.ones_like(integral), create_graph=True)[0]
            heights = intensity * torch.exp(-integral)
            expectation = (taus * heights * time_step).sum(dim=-1) #xf(x)dx

        return expectation
    