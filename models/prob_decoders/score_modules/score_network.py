import numpy as np
import torch.nn as nn
from models.embeddings import TrigonoTimeEmbedding, LinearTimeEmbedding

class ScoreNet(nn.Module):
    def __init__(self, embed_size, layer_num):
        super().__init__()
        self.embed_size=embed_size
        
        self.time_emb = TrigonoTimeEmbedding(embed_size=embed_size)
        self.h_emb = nn.Linear(embed_size, embed_size)
        self.feed_forward = nn.ModuleList([nn.Linear(embed_size, embed_size) for i in range(layer_num)])
        self.to_time = nn.Linear(embed_size, 1)
        self.activation = nn.GELU()
        
    def forward(self, x, cond):
        time_embedding = self.time_emb(x)/np.sqrt(self.embed_size)
        h_embedding = self.h_emb(cond)

        y = time_embedding + h_embedding
        for layer in self.feed_forward:
            y = layer(y)
            y = self.activation(y) + time_embedding + h_embedding
        return self.to_time(y).squeeze(dim=-1)