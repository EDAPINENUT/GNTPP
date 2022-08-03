import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def forward(self, query, key, value, mask=None, decay_weight=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        scores = torch.clip(scores, min=1e-9, max=1e9)

        if mask is not None:
            mask = mask.bool()
            scores = scores.masked_fill(mask==False, -1e9)
        
        if decay_weight is not None:
            scores = scores * decay_weight
        
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn