from math import e
import torch
from numbers import Number
import math
from torch.distributions import Distribution as TorchDistribution
from models.libs.utils import clamp_preserve_gradients
from torch.distributions.utils import broadcast_all

class ExpDecay(TorchDistribution):
    '''
    ExpDecay distribution for TorchDistribution class
    eta, beta, alpha > 0
    whose pdf: f(t) = (eta * exp(-beta * t) + alpha) * exp(-(1 - eta/ beta) * exp(-beta * t) - alpha * t)
          cdf: F(t) = 1 - exp(-(1 - eta/ beta) * exp(-beta * t) - alpha * t)
          intensity: lamda(t) = eta * exp(-beta * t) + alpha
          sampling: 
     '''
    def __init__(self, eta, beta, alpha, validate_args=False):
        assert (beta>=0).float().prod() > 0 and (eta>=0).float().prod() > 0 \
            and (alpha>=0).float().prod() > 0, ('Wrong parameter!')
        self.eta, self.beta, self.alpha = broadcast_all(eta, beta, alpha)
        if isinstance(eta, Number) and isinstance(beta, Number) and isinstance(alpha, Number) :
            batch_shape = torch.Size()
        else:
            batch_shape = self.eta.size()
        super(ExpDecay, self).__init__(batch_shape, validate_args=validate_args)

    def intensity(self, value):
        if self._validate_args:
            self._validate_sample(value)
        eta, beta, alpha = self._clamp_params()
        return eta * torch.exp(-beta * value) + alpha

    def int_intensity(self, value):
        if self._validate_args:
            self._validate_sample(value)
        eta, beta, alpha = self._clamp_params()
        a = clamp_preserve_gradients(torch.divide(eta, beta), 0, 1e7)
        return a * (1 - torch.exp(- beta * value)) + alpha * value

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        int_intensity = self.int_intensity(value)
        return 1 - torch.exp(-int_intensity)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            pass

    def log_cdf(self, value):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(value), 1e-7, 1 - 1e-7)
        return cdf.log()

    # the terms eta/beta and exp(beta * value) will explode the gradient, thus should be clamp to a small maximum!!!!!!!!

    def log_survival_function(self, value):
        # No numerically stable implementation of log survival is available for normal distribution.
        eta, beta, alpha = self._clamp_params()
        int_intensity = self.int_intensity(value)
        return - int_intensity
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        intensity = self.intensity(value)
        int_intensity = self.int_intensity(value)
        return intensity.log() - int_intensity
    

    def interval_prob(self, value):
        '''
        value: shape of: batch_size, seq_len, interval_point_num, mix_number
        return the probability of each point, shape: batch_size, seq_len, event_type_num, interval_point_num, mix_number
        '''
        eta, beta, alpha = self._clamp_params()
        eta, beta, alpha = eta[: ,: ,: , None, ...], beta[: ,: ,: , None, ...], alpha[: ,: ,: , None, ...]
        a = clamp_preserve_gradients(torch.divide(eta, beta), 0, 1e7)
        return (eta * torch.exp(-beta * value) + alpha) * torch.exp(((a - 1) * torch.exp(-beta * value) - alpha * value).clamp(max = 50))

    def _clamp_params(self):
        eta = clamp_preserve_gradients(self.eta, 1e-7, 1e7)
        beta = clamp_preserve_gradients(self.beta, 1e-7, 1e7)
        alpha = clamp_preserve_gradients(self.alpha, 1e-7, 1e7)
        return eta, beta, alpha