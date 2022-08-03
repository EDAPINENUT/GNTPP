import torch

from torch.distributions import Normal as TorchNormal

from models.libs.utils import clamp_preserve_gradients


class Normal(TorchNormal):
    def log_cdf(self, value):
        # No numerically stable implementation of log CDF is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(value), 1e-7, 1 - 1e-7)
        return cdf.log()

    def log_survival_function(self, value):
        # No numerically stable implementation of log survival is available for normal distribution.
        cdf = clamp_preserve_gradients(self.cdf(value), 1e-7, 1 - 1e-7)
        return torch.log(1.0 - cdf)

    def log_intensity(self, value):
        return self.log_prob(value) - self.log_survival_function(value)
    
    def int_intensity(self, value):
        return - self.log_survival_function(value)
    
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return torch.normal(self.loc.expand(shape), self.scale.expand(shape))