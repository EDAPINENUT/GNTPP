import torch
import torch.nn as nn
import math

class TPPWarper(nn.Module):

    def __init__(
        self,
        time_embedding: nn.Module,
        type_embedding: nn.Module,
        position_embedding: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        event_type_num: int,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        embed_size: int = 32,
        type_wise: bool = False,
        max_dt: float=50.0,
        **kwargs
    ):
        super().__init__()
        self.event_type_num = event_type_num
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.max_dt = max_dt

        self.embed_size = embed_size
        assert self.embed_size%2 == 0, ('embed_size must be an event number.')
        
        self.type_wise = type_wise
        self.type_emb = type_embedding
        self.time_emb = time_embedding
        self.position_emb = position_embedding
        
        self.encoder = encoder

        self.log_loss = decoder

    def _position_embedding(self, seq_dts):
        length = seq_dts.shape[1]
        pos = torch.arange(0, length).to(seq_dts).expand_as(seq_dts)
        pos_emb = self.position_emb(pos.long())
        return pos_emb

    def _event_embedding(self, seq_dts, seq_types):
        """
        Calculate the embedding from the sequence of events.

        Args:
            seq_dts: Time interval of events (batch_size, ... ,seq_len)
            seq_types: Sequence of event types (batch_size, ... ,seq_len)
        Returns:
            embedding: The embedding of time and event types (batch_size, ..., seq_len, embed_size)

        """
        seq_dts = self._transform_dt(seq_dts)
        type_embedding = self.type_emb(seq_types) * math.sqrt(self.embed_size//2)  #
        time_embedding = self.time_emb(seq_dts)
        embedding = torch.cat([time_embedding, type_embedding], dim=-1)
        return embedding

    def get_embedding(self, seq_dts, seq_types):
        """
        Get the embedding of given sequence of events.

        Args:
            seq_dts: Time interval of events (batch_size, ... ,seq_len)
            seq_types: Sequence of event types (batch_size, ... ,seq_len)
        Returns:
            embedding: The embedding of time and event types. The first is time, second is event (batch_size, ..., seq_len, embed_size//2)
        """
        embedding_time_type = self._event_embedding(seq_dts, seq_types)
        embedding_position = self._position_embedding(seq_dts)
        
        return embedding_time_type[...,:self.embed_size//2], embedding_time_type[...,:self.embed_size//2], embedding_position

    def _transform_dt(self, seq_dts):
        """
        Convert seqence of time intervals into normalized vector.

        Args:
            seq_dts: Time intervals of events (batch_size, ... ,seq_len)

        Returns:
            seq_dts: Normalized time intervals (batch_size, ... ,seq_len)

        """
        seq_dts = torch.log(seq_dts + 1e-8)
        seq_dts = torch.divide((seq_dts - self.mean_log_inter_time), self.std_log_inter_time)
        return seq_dts

    def _compute_similarity(self, seq_dts, seq_types):
        _, type_embedding, position_embedding = self.get_embedding(seq_dts, seq_types)
        embedding = torch.cat([type_embedding, position_embedding], dim=-1)
        embedding_norm = torch.divide(embedding, embedding.norm(dim=-1, keepdim=True) + 1e-8)
        similarity = torch.matmul(embedding_norm, embedding_norm.transpose(-1,-2))
        return similarity
    
    def get_similarity(self):
        em = self.type_emb.weight[:-1, :]
        em_unit = torch.divide(em, em.norm(dim=-1, keepdim=True))
        similarity = torch.mm(em_unit, em_unit.transpose(0,1)).clamp(max=1, min=-1)
        return similarity
    
    def plot_similarity(self, file_name):
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams.update({'font.size': 20})
        fig, ax = plt.subplots(figsize=(16,15))
        similarity = self.get_similarity().detach().cpu().numpy()
        im = ax.matshow(similarity, cmap=plt.get_cmap('RdBu'), vmin=-1, vmax=1)
        plt.axis('off')
        cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0,0.03,ax.get_position().height])

        fig.colorbar(im, cax=cax)
        plt.savefig(file_name + '.png')
        plt.show()


    def forward(self, seq_dts, seq_types, lag_matrixes=None, *args):
        
        event_embedding = self._event_embedding(seq_dts, seq_types)
        similarity_matrixes = self._compute_similarity(seq_dts, seq_types)
        
        self.history_embedding = self.encoder(seq_types, event_embedding, lag_matrixes, similarity_matrixes)
        
        if self.type_wise == True:
            self.history_embedding = self.history_embedding.unsqueeze(dim=-2).expand(-1,-1,self.event_type_num,-1)
        else:
            self.history_embedding = self.history_embedding.unsqueeze(dim=-2)
            
        return self.history_embedding


    def compute_loss(self, batch, *args):
 
        history_embedding = self.learn(batch)
        
        seq_dts, seq_types, seq_onehots = batch.out_dts, batch.out_types, batch.out_onehots
        
        log_loss, mark_loss = self.log_loss(seq_dts, seq_types, seq_onehots, history_embedding)
        return log_loss + mark_loss 

    def compute_nll(self, batch, *args):
        
        history_embedding = self.history_embedding
        seq_dts, seq_types, seq_onehots = batch.out_dts, batch.out_types, batch.out_onehots
        
        return self.log_loss.compute_nll(seq_dts, seq_onehots, history_embedding)

    def compute_ce(self, batch, *args):
        
        history_embedding = self.history_embedding
        seq_dts, seq_types = batch.out_dts, batch.out_types
        
        return self.log_loss.compute_ce(history_embedding, seq_types)


    def learn(self, batch, *args):
        seq_dts, seq_types, lag_matrixes= batch.in_dts, batch.in_types, batch.lag_matrixes ##################
        return self.forward(seq_dts, seq_types, lag_matrixes, *args)

    def evaluate(self, batch, *args):
        seq_dts, seq_types, lag_matrixes= batch.in_dts, batch.in_types, batch.lag_matrixes
        return self.forward(seq_dts, seq_types, lag_matrixes, *args)
    
    def sample(self, batch, sample_num, max_t=None, reforward=False, *args):
        seq_dts, seq_types, lag_matrixes, seq_onehots = \
            batch.in_dts, batch.in_types, batch.lag_matrixes, batch.out_onehots
        if reforward == True:
            self.forward(seq_dts, seq_types, lag_matrixes, *args)
        return self.log_loss.t_sample(self.history_embedding, seq_onehots, sample_num, max_t)
    
    def predict_event_time(self, max_t, resolution=500):
        max_t = max_t.item() if torch.is_tensor(max_t) else max_t
        history_embedding = self.history_embedding
        return self.log_loss.inter_time_dist_pred(history_embedding, max_t, resolution)
    
    def predict_event_type(self):
        if hasattr(self.log_loss, 'mark_logits'):
            return self.log_loss.mark_logits
        else:
            return None
    
    def cumulative_risk_func(self, batch, sample_num=200, steps=20, reforward=False, *args):
        seq_dts, seq_types, lag_matrixes, seq_onehots = \
            batch.in_dts, batch.in_types, batch.lag_matrixes, batch.out_onehots
        if reforward == True:
            self.forward(seq_dts, seq_types, lag_matrixes, *args)
        return self.log_loss.cumulative_risk_func(self.history_embedding, seq_dts, sample_num, self.max_dt, steps)
        