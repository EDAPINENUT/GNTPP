import torch
import torch
import torch.nn as nn
from .base_intensity_dec import BaseIntensityDecoder
from models.basic_layers.normlayer import LayerNorm
import torch.distributions as D
from .distributions import Weibull, MixtureSameFamily, TransformedDistribution

class WeibullMixtureDistribution(TransformedDistribution):
    """
    Mixture of Weibull distributions.

    Args:
        locs: Location parameters of the component distributions,
            shape (batch_size, seq_len, num_mix_components)
        r_betas: betas of the component Weibull distributions, which is not all positive,
            shape (batch_size, seq_len, num_mix_components)
        r_etas: etas of the component Weibull distributions, which is not all positive,
            shape (batch_size, seq_len, num_mix_components)
        mean_log_inter_time: Average log-inter-event-time
        std_log_inter_time: Std of log-inter-event-times
    """
    def __init__(
        self,
        r_betas: torch.Tensor,
        r_etas: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0
    ):
        mixture_dist = D.Categorical(logits=log_weights)
        component_dist = Weibull(beta=r_betas.abs(), eta=r_etas.abs())
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        if mean_log_inter_time == 0.0 and std_log_inter_time == 1.0:
            transforms = []
        else:
            transforms = [D.AffineTransform(loc=mean_log_inter_time, scale=std_log_inter_time)]
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        super().__init__(GMM, transforms)

    def mean(self, *args) -> torch.Tensor:
        """
        Compute the expected value of the distribution.

        Returns:
            mean: Expected value, shape (batch_size, seq_len)
        """
        eta = self.base_dist._component_distribution.eta
        beta = self.base_dist._component_distribution.beta
        log_weights = self.base_dist._mixture_distribution.logits
        mean = (-eta.log() + torch.lgamma(1 + 1/beta) + log_weights).exp().sum(dim=-1)
        return mean

    def sample(self, sample_num=100):
            
        sample = self.base_dist.sample(sample_num=sample_num)
        
        a = self.std_log_inter_time
        b = self.mean_log_inter_time

        sample = sample * a + b
        mask = torch.ones_like(sample)
        return sample, mask

class WeibMix(BaseIntensityDecoder):
    """

    The distribution of the inter-event times given the history is modeled with a Weibull mixture distribution.

    Args:
        embed_dim: Dimension of the embedding vectors
        layer_num: Number of layers for non-linear transformations
        event_type_num: Number of event type to consturct the number of intensity functions
        mean_log_inter_time: Average log-inter-event-time
        std_log_inter_time: Std of log-inter-event-times
        num_mix_components: Number of mixture components in the inter-event time distribution.
    """

    def __init__(
        self,
        embed_size: int,
        layer_num: int,
        event_type_num: int,
        mean_log_inter_time: float=0.0,
        std_log_inter_time: float=1.0,
        num_mix_components: int = 16,
        *args,
        **kwargs
        ):
        super().__init__(embed_size, layer_num, event_type_num, mean_log_inter_time, std_log_inter_time, **kwargs)
        self.num_mix_components = num_mix_components
        linear = nn.ModuleList([nn.Linear(self.embed_size, 3*num_mix_components, bias=True)])
        for i in range(layer_num - 1):
            linear.append(nn.GELU())
            linear.append(nn.Linear(3*num_mix_components, 3*num_mix_components, bias=True))
        self.linear = nn.Sequential(*linear)
        self.norm = LayerNorm(3*num_mix_components)

    def get_inter_time_dist(self, history_embedding: torch.Tensor) -> torch.distributions.Distribution:
        """
        Get the distribution over inter-event times given the history_embedding.

        Args:
            history_embedding: history_embedding vector used to condition the distribution of each event,
                shape (baWtch_size, seq_len, embed_dim)

        Returns:
            dist: Distribution over inter-event times, has batch_shape (batch_size, seq_len)

        """
        raw_params = self.norm(self.linear(history_embedding))  # (batch_size, seq_len, 3 * num_mix_components)
        # Slice the tensor to get the parameters of the mixture
        r_betas = raw_params[..., :self.num_mix_components]
        r_etas = raw_params[..., self.num_mix_components: (2 * self.num_mix_components)]
        log_weights = raw_params[..., (2 * self.num_mix_components):]

        log_weights = torch.log_softmax(log_weights, dim=-1)

        return WeibullMixtureDistribution(
            r_betas=r_betas,
            r_etas=r_etas,
            log_weights=log_weights
        )

