from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(
        self,
        event_type_num: int,
        input_size: int = 32, 
        embed_size: int = 32, 
        layer_num: int = 1, 
        drop_ratio: float = 0.1,
        activation: str= 'tanh',
    ):  

        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.layer_num = layer_num
        self.event_type_num = event_type_num
        self.drop_ratio = drop_ratio
        self.dropout = nn.Dropout(drop_ratio)
        self.activation = getattr(torch, activation)
    

    def _hist_encoding(self, seq_types, embedding, *args, **kwargs):
        raise NotImplementedError()

