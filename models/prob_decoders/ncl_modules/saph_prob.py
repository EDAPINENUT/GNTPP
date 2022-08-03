from .base_ncl_decoder import BaseNCLDecoer
import torch.nn as nn
import torch.nn.functional as F
from .integral import * 
import math

def softplus(x, beta):
    # hard thresholding at 20
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class SAHPDecoder(BaseNCLDecoer):
    def __init__(self,
                 embed_size,
                 layer_num,
                 event_type_num,
                 mean_log_inter_time: float=0.0,
                 std_log_inter_time: float=1.0,
                 *args,
                 **kwargs):
        super(SAHPDecoder, self).__init__(
                        embed_size,
                        layer_num,
                        event_type_num,
                        mean_log_inter_time,
                        std_log_inter_time,
                        *args,
                        **kwargs)


        self.start_layer = nn.Sequential(
            nn.Linear(embed_size, self.event_type_num, bias=True),
            nn.GELU()
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(embed_size, self.event_type_num, bias=True),
            nn.GELU()
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(embed_size, self.event_type_num, bias=True)
            ,nn.Softplus(beta=10.0)
        )

        self.intensity_layer = nn.Sequential(
            nn.Softplus(beta=1.)
        )

    def intensity_func(self, temp_time, temp_hid):
        converge_hid = temp_hid[:,:,0]
        start_hid = temp_hid[:,:,1]
        omega_hid = temp_hid[:,:,2]
        intensity = converge_hid + (start_hid - converge_hid) * torch.exp(- omega_hid * temp_time)
        return self.intensity_layer(intensity)

    def his_to_param(self, history_embedding):
        batch_size, seq_len, event_num, embed_size = history_embedding.shape
        converge_hid = self.converge_layer(history_embedding)
        start_hid = self.start_layer(history_embedding)
        omega_hid = self.decay_layer(history_embedding)

        temp_hid = torch.cat([converge_hid, start_hid, omega_hid], dim=2)
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


