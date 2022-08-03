from math import e
import torch
from numbers import Number
import math
from torch.distributions import Distribution as TorchDistribution
from models.libs.utils import clamp_preserve_gradients
from torch.distributions.utils import broadcast_all

class LogCauchy(TorchDistribution):
    '''
    LogCauchy distribution for TorchDistribution class
    whose pdf: f(t) = 1 / (t * pi) *(sigma / ((log(t) - mu)**2 + sigma ** 2))
          cdf: F(t) = 1 / pi * arctan((log(t) - mu) / sigma) + 1/2
          intensity: lamda(t) = f(1) / [1 - F(t)]
          sampling: 
     '''
    def __init__(self, mu, sigma, validate_args=False):
        assert (sigma>=0).float().prod() > 0, \
            ('Wrong parameter!')
        self.mu, self.sigma = broadcast_all(mu, sigma)
        if isinstance(mu, Number) and isinstance(sigma, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.mu.size()
        super(LogCauchy, self).__init__(batch_shape, validate_args=validate_args)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma = self._clamp_params()
        return 1 / math.pi * torch.arctan((torch.log(value) - mu) / sigma) + 1/2

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = self.mu.new(shape).cauchy_()
        return (self.mu + eps * self.sigma).exp()

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.
        """
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_cdf(self, value):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(value), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, value):
        cdf = clamp_preserve_gradients(self.cdf(value), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        mu, sigma = self._clamp_params()
        return torch.log(1 / (value * math.pi) * (sigma / ((torch.log(value) - mu)**2 + sigma ** 2)))

    def interval_prob(self, value):
        '''
        value: shape of: batch_size, seq_len, interval_point_num, mix_number
        return the probability of each point, shape: batch_size, seq_len, event_type_num, interval_point_num, mix_number
        '''
        mu, sigma = self._clamp_params()
        mu, sigma = mu[: ,: ,: , None, ...], sigma[: ,: ,: , None, ...]
        return 1 / (value * math.pi) * (sigma / ((torch.log(value) - mu)**2 + sigma ** 2))
    
    def _clamp_params(self):
        mu = clamp_preserve_gradients(self.mu, 1e-7, 1e7) + 1e-7
        sigma = clamp_preserve_gradients(self.sigma, 1e-7, 1e7)
        return mu, sigma