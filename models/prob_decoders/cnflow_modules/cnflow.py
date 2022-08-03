from .ode_func import *
from .unit_flow import *
from models.libs.sampling import *
from ..base_prob_dec import BaseProbDecoder

class CNFDecoder(BaseProbDecoder):

    def __init__(self,
                 embed_size: int,
                 layer_num: int,
                 event_type_num: int,
                 mean_log_inter_time: float=0.0,
                 std_log_inter_time: float=1.0,
                 *args,
                 **kwargs):
        super().__init__(embed_size, layer_num, event_type_num, mean_log_inter_time, std_log_inter_time, *args, **kwargs)

        self.flow = build_flow(
            embed_size=embed_size,
            layer_num=layer_num,
        )

    def get_t(self, z, history_embedding):
        t = self.flow(
            z,
            history_embedding = history_embedding,
            reverse = True,
        )
        return t

    def get_z(self, t, history_embedding):
        z = self.flow(
            t,
            history_embedding = history_embedding,
            reverse = False
        )
        return z

    def compute_nll(self, seq_dts, seq_onehots, history_embedding, *args):
        batch_size, seq_length, event_num, embed_size = history_embedding.shape
        seq_dts = seq_dts.clamp(1e-8)
        seq_dts_expand = self.normalize_dt(seq_dts).unsqueeze(dim=-1).expand(-1, -1, event_num)
        
        shape = history_embedding.shape[:-1]
        z, delta_logpz = self.flow(
            seq_dts_expand,
            history_embedding = history_embedding,
            logpt=torch.zeros(shape + (1,)).to(seq_dts_expand)
        )
        z, delta_logpz = z.reshape(*seq_dts_expand.shape), delta_logpz.reshape(shape + (1,))
        log_pz = standard_normal_logprob(z) * seq_onehots
        delta_logpz = delta_logpz.squeeze(dim=-1) * seq_onehots
        normalize_logpt = (self.std_log_inter_time.log() + seq_dts.log().unsqueeze(dim=-1)) * seq_onehots
        log_pd = -log_pz + delta_logpz + normalize_logpt
        return log_pd.sum()
    

    def forward(self, seq_dts, seq_types, seq_onehots, history_embedding, *args):
        log_loss = self.compute_nll(seq_dts, seq_onehots, history_embedding)
        mark_logits = self.compute_ce(history_embedding, seq_types)
        
        return log_loss, mark_logits

    def t_sample(self, history_embedding, seq_onehots=None, sample_num=100, *args):
        with torch.no_grad():
            batch_size, seq_length, event_num, embed_size = history_embedding.shape
            sample_shape = torch.Size((batch_size * sample_num, seq_length, event_num))
            history_embedding_expand = history_embedding[:,None,...]\
                                .expand(-1,sample_num,-1,-1,-1)\
                                .reshape((batch_size * sample_num, seq_length, event_num, embed_size))
                                
            z = torch.randn(sample_shape).to(history_embedding)

            sample_intervals = self.flow(
                z,
                history_embedding_expand,
                # integration_times=torch.linspace(0,1,10),
                reverse = True
            )
                    
            sample_intervals = sample_intervals.reshape(batch_size, sample_num, seq_length, event_num)
            sample_intervals = self.unnormalize_dt(sample_intervals)
            
            mask = torch.ones_like(sample_intervals)
        return sample_intervals, mask
    
    def inter_time_dist_pred(self, history_embedding, max_t, resolution):
        # using MC get mean, sample number = resolution
        interval, _ = self.t_sample(history_embedding, sample_num=resolution)
        
        interval = interval.mean(dim=1)
        return interval

    def cumulative_risk_func(self, history_embedding, dt, sample_num=200, max_dt=5, steps=20):
        return self.empirical_cumulative_risk_func(history_embedding, dt, sample_num=sample_num, max_dt=max_dt, steps=steps)
