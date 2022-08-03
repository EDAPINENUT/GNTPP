import torch
from torch import nn
from .encoder import Encoder


class LSTMEnc(Encoder):
    def __init__(
        self,
        event_type_num: int,
        embed_size: int = 16, 
        layer_num: int = 1, 
        drop_ratio: float = 0.1, 
        activation='tanh',
        **kwargs
    ):
        super().__init__(event_type_num, embed_size, embed_size, layer_num, drop_ratio, activation)
        self.recurrent_nn = nn.LSTM(input_size = embed_size, hidden_size = embed_size, num_layers = layer_num, batch_first=True)
        
    def forward(self, seq_types, embedding, lag_matrixes=None, similarity_matrixes=None):
        """
        embedding: shape (batch_size, seq_length, emb_dim)
        """
        
        return self._hist_encoding(seq_types, embedding, lag_matrixes, similarity_matrixes)
        

    def _hist_encoding(self, seq_types, embedding, lag_matrixes=None, similarity_matrixes=None):
        batch_size, seq_len, emb_size = embedding.shape
        history_embdedding, _ = self.recurrent_nn(embedding)
        return history_embdedding