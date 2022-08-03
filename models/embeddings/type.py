import torch.nn as nn


class TypeEmbedding(nn.Embedding):
    def __init__(self, event_type_num, embed_size, padding_idx, **kwargs):
        super().__init__(event_type_num + 1, embed_size, padding_idx=padding_idx)