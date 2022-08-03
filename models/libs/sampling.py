import math
import torch 

def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def rejection_sampling(p, f_sampler, f, upperbound, shape, type_mask=True, dim=0, device='cuda', tolerance=100):
    
    sample = f_sampler(shape).to(device)
    mask = torch.zeros_like(sample).bool()
    samples = [sample]
    masks = mask
    sample_num = mask.sum(dim=dim)
    i=1
    
    while ((sample_num <= 2)*type_mask).bool().any() and i<tolerance:
    
        U = torch.rand_like(sample)
        F = f(sample)
        P = p(sample)
        mask = (U <= torch.reciprocal(F) * P / upperbound)
        sample_num += mask.sum(dim=dim)
        masks = torch.cat([masks, mask],dim=dim)
        del F, U, P
        with torch.cuda.device('cuda:{}'.format(device.index)):
            torch.cuda.empty_cache()
        samples.append(sample)
        i += 1
    return torch.cat(samples, dim=dim), masks
    