import torch
from torch.distributions import MixtureSameFamily as TorchMixtureSameFamily, categorical


class MixtureSameFamily(TorchMixtureSameFamily):
    def log_cdf(self, value):
        value = self._pad(value)
        log_cdf_x = self.component_distribution.log_cdf(value)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_cdf_x + mix_logits, dim=-1)

    def log_survival_function(self, value):
        value = self._pad(value)
        log_sf_x = self.component_distribution.log_survival_function(value)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_sf_x + mix_logits, dim=-1)
    
    def log_prob(self, value):
        value = self._pad(value)
        log_prob_x = self.component_distribution.log_prob(value)
        mix_logits = self.mixture_distribution.logits
        return torch.logsumexp(log_prob_x + mix_logits, dim=-1)

    def log_intensity(self, value):
        return self.log_prob(value) - self.log_survival_function(value)
    
    def sample(self, sample_num=100):
        comp_value = self.component_distribution.sample(sample_shape=(sample_num,))
        category = self.mixture_distribution.sample(sample_shape=(sample_num,)).unsqueeze(dim=-1)
        return torch.gather(comp_value, dim=-1, index=category).squeeze(dim=-1).transpose(0,1)