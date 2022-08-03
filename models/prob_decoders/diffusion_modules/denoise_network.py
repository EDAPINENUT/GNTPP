import torch.nn as nn
from models.embeddings import TrigonoTimeEmbedding, LinearTimeEmbedding
from .diffusion_embedding import DiffusionEmbedding
import numpy as np

class DenoiseNet(nn.Module):
    def __init__(self, embed_size, layer_num):
        super().__init__()
        self.embed_size=embed_size
        
        self.time_emb = TrigonoTimeEmbedding(embed_size=embed_size)
        self.h_emb = nn.Linear(embed_size, embed_size)
        self.feed_forward = nn.ModuleList([nn.Linear(embed_size, embed_size) for i in range(layer_num)])
        self.to_time = nn.Linear(embed_size, 1)
        self.activation = nn.GELU()
        self.diffusion_time_emb = DiffusionEmbedding(embed_size=embed_size)
        
    def forward(self, x, t, cond):
        time_embedding = self.time_emb(x.squeeze(dim=-1))/np.sqrt(self.embed_size)
        cond = self.h_emb(cond)
        b, *_ = time_embedding.shape
        
        diff_time_embedding = self.diffusion_time_emb(t)\
                              .reshape(b, *(1,) * (len(time_embedding.shape) - 2), self.embed_size)\
                              .expand_as(time_embedding)
        
        y = time_embedding + diff_time_embedding + cond
        for layer in self.feed_forward:
            y = layer(y)
            y = self.activation(y) + time_embedding + diff_time_embedding + cond
        return self.to_time(y)
        
        