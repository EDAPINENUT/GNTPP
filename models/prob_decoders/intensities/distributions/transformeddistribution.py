from torch.distributions import TransformedDistribution as TorchTransformedDistribution
from torch.distributions.utils import _sum_rightmost


class TransformedDistribution(TorchTransformedDistribution):
    def __init__(self, base_distribution, transforms, validate_args=None):
        super().__init__(base_distribution, transforms, validate_args=validate_args)
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        self.sign = int(sign)

    def log_cdf(self, value):
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)

        if self.sign == 1:
            return self.base_dist.log_cdf(value)
        else:
            return self.base_dist.log_survival_function(value)

    def log_survival_function(self, value):
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)

        if self.sign == 1:
            return self.base_dist.log_survival_function(value)
        else:
            return self.base_dist.log_cdf(value)

    def log_intensity(self, value):
        event_dim = len(self.event_shape)
        log_intensity = 0.0
        y = value
        for transform in reversed(self.transforms):
            x = transform.inv(y)
            log_intensity = log_intensity - _sum_rightmost(transform.log_abs_det_jacobian(x, y),
                                                    event_dim - transform.event_dim)
            y = x

        log_intensity = log_intensity + _sum_rightmost(self.base_dist.log_intensity(y),
                                                event_dim - len(self.base_dist.event_shape))
        return log_intensity
    
    def int_intensity(self, value):
        return - self.log_survival_function(value)