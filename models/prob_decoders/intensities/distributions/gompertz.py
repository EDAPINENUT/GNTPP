from math import e
import torch
from numbers import Number
import math
from torch.distributions import Distribution as TorchDistribution
from models.libs.utils import clamp_preserve_gradients
from torch.distributions.utils import broadcast_all

class Gompertz(TorchDistribution):
    '''
    Gompertz distribution for TorchDistribution class
    whose pdf: f(t) = beta * eta * exp(eta + beta * t - eta * exp(beta * t))
          cdf: F(t) = 1 - exp(- eta * (exp(beta * t) - 1))
          intensity: lamda(t) = eta * beta * exp(beta * t)
          sampling: u ~ U[0,1]  
                    -ln(u) ~ exp(1) 
                    1/beta * ln(1 - 1 / eta * ln(u) ) ~ Gompt(eta, beta)
     '''
    
    def __init__(self, eta, beta, validate_args=False):
        assert (beta>=0).float().prod() > 0 and (eta>=0).float().prod() > 0, \
            ('Wrong parameter!')
        self.eta, self.beta = broadcast_all(eta, beta)
        if isinstance(eta, Number) and isinstance(beta, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.eta.size()
        super(Gompertz, self).__init__(batch_shape, validate_args=validate_args)

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        eta, beta = self._clamp_params()
        b = clamp_preserve_gradients(beta * value, 0, 5e1)
        return 1 - torch.exp(- eta * (torch.exp(b) - 1.0))

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            u = torch.rand(shape).to(self.eta).clamp(max=1-1e-8, min=1e-8)
            eta, beta = self._clamp_params()
            eta, beta = eta.expand(shape), beta.expand(shape)
            return torch.divide(torch.log(1.0 - torch.reciprocal(eta) * torch.log(u)), beta)

    def log_cdf(self, value):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(value), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, value):
        eta, beta = self._clamp_params()
        b = clamp_preserve_gradients(beta * value, 0, 5e1)
        return - eta * (torch.exp(b) - 1.0)
    
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        eta, beta = self._clamp_params()
        log_eta = math.log(eta) if isinstance(eta, Number) else eta.log()
        log_beta = math.log(beta) if isinstance(beta, Number) else beta.log()
        b = clamp_preserve_gradients(beta * value, 0, 5e1)
        return log_eta + log_beta + eta + b - eta * b.exp()
    
    def log_intensity(self, value):
        '''
        log of the intensity function
        '''
        if self._validate_args:
            self._validate_sample(value)
        eta, beta = self._clamp_params()
        b = clamp_preserve_gradients(beta * value, 0, 5e1)
        log_eta = math.log(eta) if isinstance(eta, Number) else eta.log()
        log_beta = math.log(beta) if isinstance(beta, Number) else beta.log()

        return log_eta + log_beta + b
    
    def int_intensity(self, value):
        '''
        Integral of the intensity function
        '''

        return -self.log_survival_function(value)

    def interval_prob(self, value):
        '''
        value: shape of: batch_size, seq_len, interval_point_num, mix_number
        return the probability of each point, shape: batch_size, seq_len, event_type_num, interval_point_num, mix_number
        '''
        eta, beta = self._clamp_params()
        eta, beta = eta[: ,: ,: , None, ...], beta[: ,: ,: , None, ...]
        b = clamp_preserve_gradients(beta * value, 0, 5e1)
        return torch.exp(beta * value + eta - eta * torch.exp(b)) * eta * beta

    def _clamp_params(self):
        eta = clamp_preserve_gradients(self.eta, 1e-7, 1e7)
        beta = clamp_preserve_gradients(self.beta, 1e-7, 1e7)
        return eta, beta