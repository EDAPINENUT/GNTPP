import torch.nn as nn
import torch

class TrigonoTimeEmbedding(nn.Module):
    def __init__(self, embed_size, **kwargs):
        super().__init__()
        assert embed_size%2 == 0 
        
        self.Wt = nn.Linear(1, embed_size // 2, bias=False)

    def forward(self, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        pe_sin = torch.sin(phi)
        pe_cos = torch.cos(phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe

class LinearTimeEmbedding(nn.Module):
    def __init__(self, embed_size, **kwargs):
        super().__init__()
        self.Wt = nn.Linear(1, embed_size, bias=False)
    
    def forward(self, interval):
        emb = self.Wt(interval.unsqueeze(-1))
        return emb