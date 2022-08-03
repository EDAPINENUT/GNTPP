import torch
import torch.nn as nn
from torch.nn.functional import one_hot

class WasDiscriminator(nn.Module):
    def __init__(self, 
                 embed_size,
                 layer_num,
                 event_type_num,
                 **kwargs) -> None:
        super().__init__()

        self.layer_num = layer_num
        self.embed_size = embed_size
        self.event_type_num = event_type_num
        
        self.network_t = nn.Linear(1, self.embed_size)
        self.network_h = nn.Linear(self.embed_size, self.embed_size)
        
        network = []
        for i in range(layer_num):
            network.append(nn.Linear(self.embed_size, self.embed_size))
        self.out_network = nn.Linear(self.embed_size, 1)            
        self.network = nn.ModuleList(network)
        self.activation = nn.GELU()   
        
    def mlp_transform(self, time, h):
        t_emb = self.network_t(time.unsqueeze(dim=-1))
        h_emb = self.network_h(h)
        emb = self.activation(t_emb + h_emb)
        for linear in self.network:
            emb = self.activation(linear(emb))
        return self.out_network(emb)
    
    def lipschitz_loss(self, sample_t, true_t, out_sample_emb, out_true_emb):
        lip = torch.divide((out_sample_emb - out_true_emb).abs(), (sample_t-true_t).unsqueeze(dim=-1).abs()+1e-8)
        return (lip - 1).abs().sum()
    
    def g_loss(self, sample_t, batch, hist_embedding):
        true_t = batch.out_dts
        true_t = true_t[...,None]
        one_hot = batch.out_onehots
        one_hot = one_hot[...,None]
        if len(sample_t.shape) - len(true_t.shape) == 1:
            true_t = true_t[:, None,...].expand_as(sample_t)
            hist_embedding = hist_embedding[:,None,...].expand(sample_t.shape + (-1,))
            one_hot = one_hot[:,None,...].expand(sample_t.shape[:-1] + (-1,-1,))
        out_sample_emb = self.mlp_transform(sample_t, hist_embedding)
        out_true_emb = self.mlp_transform(true_t, hist_embedding)
        return ((out_sample_emb - out_true_emb) * one_hot).abs().sum()
    
    def d_loss(self, sample_t, batch, hist_embedding, nu=1.0):
        true_t = batch.out_dts
        true_t = true_t[...,None]
        one_hot = batch.out_onehots
        one_hot = one_hot[...,None]
        
        if len(sample_t.shape) - len(true_t.shape) == 1:
            true_t = true_t[:, None,...].expand_as(sample_t)
            hist_embedding = hist_embedding[:,None,...].expand(sample_t.shape + (-1,))
            one_hot = one_hot[:,None,...].expand(sample_t.shape[:-1] + (-1,-1,))
            
        out_sample_emb = self.mlp_transform(sample_t, hist_embedding)
        out_true_emb = self.mlp_transform(true_t, hist_embedding)
        
            
        if nu == 0:
            return -((out_sample_emb - out_true_emb)* one_hot).abs().sum()
        
        return -((out_sample_emb - out_true_emb)* one_hot).abs().sum() + \
            nu * self.lipschitz_loss(sample_t, true_t, out_sample_emb, out_true_emb)
    
    