from .base_ncl_decoder import BaseNCLDecoer
import torch.nn as nn
import torch.nn.functional as F
from .integral import * 

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))


class THPDecoder(BaseNCLDecoer):
    def __init__(self,
                 embed_size,
                 layer_num,
                 event_type_num,
                 mean_log_inter_time: float=0.0,
                 std_log_inter_time: float=1.0,
                 *args,
                 **kwargs):
        super(THPDecoder, self).__init__(
                        embed_size,
                        layer_num,
                        event_type_num,
                        mean_log_inter_time,
                        std_log_inter_time,
                        *args,
                        **kwargs)

        self.param_linear = nn.Linear(embed_size, event_type_num)

    def intensity_func(self, temp_time, temp_hid, alpha=-0.1, beta=1.0):
        # temp_time = temp_time / torch.cumsum(temp_time, dim=1)
        # The renormalization does not contribute to the performance gain.
        return softplus(temp_hid[:,:,0] + alpha * temp_time, beta)

    def his_to_param(self, history_embedding):
        batch_size, seq_len, event_num, embed_size = history_embedding.shape
        temp_hid = softplus(self.param_linear(history_embedding), beta=1.0)
        return temp_hid

    def compute_nll(self, seq_dts, seq_onehots, history_embedding, *args):
        batch_size, seq_len, event_num, embed_size = history_embedding.shape
        seq_dts = seq_dts.clamp(1e-8)
        tau = seq_dts.unsqueeze(dim=-1).detach().expand(batch_size, seq_len, event_num)
        
        temp_hid = self.his_to_param(history_embedding)
        
        intensity = self.intensity_func(tau, temp_hid)
        log_intensity = ((intensity + 1e-8).log() * seq_onehots).sum(dim=-1)
        
        integral = compute_integral_unbiased(self.intensity_func, seq_dts, temp_hid, seq_onehots)
        
        log_loss =  -log_intensity + integral
        log_loss = log_loss * seq_onehots.sum(dim=-1)

        return log_loss.sum()


