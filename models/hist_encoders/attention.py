import torch
from torch import nn
from .encoder import Encoder
from models.basic_layers import *
from models.libs.order import MaskBatch, TimeDecayer

class AttentionEnc(Encoder):
    def __init__(
        self,
        event_type_num: int,
        input_size: int = 32, 
        embed_size: int = 32, 
        layer_num: int = 1, 
        drop_ratio: float = 0.1, 
        activation='tanh',
        attention_heads: int = 4,
        *args,
        **kwargs
    ):

        super().__init__(event_type_num, input_size, embed_size, layer_num, drop_ratio, activation)
        self.atten_heads = attention_heads
        self.input_sublayer = SublayerConnection(size=self.embed_size, dropout=drop_ratio)
        self.output_sublayer = SublayerConnection(size=self.embed_size, dropout=drop_ratio)
        self.feed_forward = PositionwiseFeedForward(d_model=self.embed_size, d_ff=self.embed_size * 4, dropout=drop_ratio)
        attention = [MultiHeadedAttention(h=self.atten_heads, d_model=self.embed_size) for i in range(self.layer_num)]
        self.attention = nn.Sequential(*attention)
        
        self.batch_masker = MaskBatch(pad_index=event_type_num)
        self.time_decayer = TimeDecayer(heads=attention_heads)
        
    def forward(self, seq_types, embedding, lag_matrixes=None, similarity_matrixes=None):
        """
        embedding: shape (batch_size, seq_length, emb_dim)
        """
        
        return self._hist_encoding(seq_types, embedding, lag_matrixes, similarity_matrixes)
        

    def _hist_encoding(self, seq_types, embedding, lag_matrixes=None, similarity_matrixes=None):
        
        batch_size, seq_length, embed_size = embedding.shape
        
        src_mask = self.batch_masker.make_std_mask(seq_types).reshape(-1, seq_length, seq_length)
        
        if lag_matrixes is not None:
            time_decay = self.time_decayer(lag_matrixes)
        else:
            time_decay = None
        
        decay_weight = time_decay * similarity_matrixes.unsqueeze(dim=1) \
            if (similarity_matrixes is not None) and (time_decay is not None) else time_decay
        
        x = embedding

        for i in range(self.layer_num):
            x = self.input_sublayer(x, 
                                    lambda _x: self.attention[i]\
                                    .forward(_x, _x, _x, mask=src_mask, decay_weight=decay_weight))
            x = self.dropout(self.output_sublayer(x, self.feed_forward))
        
        history_embedding = x.reshape(batch_size, seq_length, embed_size)
        return history_embedding
        
