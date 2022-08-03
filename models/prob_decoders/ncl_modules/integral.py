import torch 
import torch.nn.functional as F

    
def compute_integral_unbiased(intensity_func, seq_dts, temp_hid, seq_onehots):
    """ Log-likelihood of non-events, using Monte Carlo integration. """

    num_samples = 100

    diff_time = seq_dts.unsqueeze(dim=-1) * seq_onehots
    temp_time = diff_time.unsqueeze(-1) * \
                torch.rand([*diff_time.size(), num_samples], device=temp_hid.device)

    all_lambda = intensity_func(temp_time, temp_hid.unsqueeze(dim=-1))
    all_lambda = torch.sum(all_lambda, dim=-1) / num_samples

    unbiased_integral = all_lambda * diff_time
    return unbiased_integral.sum(dim=-1)

def inter_time_dist_pred(intensity_func, temp_hid, max_t=40, resolution=500):
    batch_size, seq_len, param_num, event_type_num = temp_hid.shape
    time_step = max_t / resolution
    x_axis = torch.linspace(0, max_t, resolution).to(temp_hid)
    taus = x_axis[None,None,None,:].\
        expand(batch_size, seq_len, event_type_num, -1).detach()

    intensity = intensity_func(taus, temp_hid.unsqueeze(dim=-1))
    integral = torch.cumsum(time_step * intensity, dim=-1)
    heights = intensity * torch.exp(-integral)
    expectation = (taus * heights * time_step).sum(dim=-1)
    return expectation

