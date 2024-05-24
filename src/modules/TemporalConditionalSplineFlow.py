import lightning.pytorch as pl
import torch.nn as nn
import torch

import pyro.distributions as distrib
import torch.distributions as td
from pyro.distributions.conditional import ConditionalTransform
from pyro.distributions.transforms import ComposeTransform
from pyro.distributions.transforms.spline import ConditionalSpline, Spline

from src.modules.TemporalHyperNet import TemporalHyperNet
from pyro.distributions import constraints
from pyro.distributions.torch_transform import TransformModule
from torch import nn


class AffineDiagonalPyro(TransformModule):
    """
    This creates a diagonal affine transformation compatible with pyro transforms
    """

    domain = constraints.real
    codomain = constraints.real
    bijective = True

    def __init__(self, input_dim: int):
        super().__init__(cache_size=1)
        self.dim = input_dim
        self.a = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            x: tensor with shape [batch, input_dim]
        Returns:
            Transformed inputs
        """
        return self.a.exp().unsqueeze(0) * x + self.b.unsqueeze(0)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Reverse method
        Args:
            y: tensor with shape [batch, input]
        Returns:
            Reversed input
        """
        return (-self.a).exp().unsqueeze(0) * (y - self.b.unsqueeze(0))

    def log_abs_det_jacobian(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _ = x, y
        return self.a.unsqueeze(0)


class TemporalConditionalSplineFlow(pl.LightningModule):

    def __init__(self,
                 hypernet
                 ):
        super().__init__()

        self.hypernet = hypernet
        self.num_bins = self.hypernet.num_bins
        self.order = self.hypernet.order

    def log_prob(self, 
                 X_input: torch.Tensor, 
                 X_history: torch.Tensor, 
                 A: torch.Tensor,
                 embeddings: torch.Tensor=None):
        """
        Args:
            X_input: input data of shape (batch, num_nodes, data_dim)
            X_history: input data of shape (batch, lag, num_nodes, data_dim)
            A: adjacency matrix of shape (batch, lag+1, num_nodes, num_nodes)
            embeddings: embeddings (batch, lag+1, num_nodes, embedding_dim)
        """

        assert len(X_history.shape) == 4

        batch, lag, num_nodes, data_dim = X_history.shape

        # if not self.trainable_embeddings:
        self.transform = nn.ModuleList(
            [
                ConditionalSpline(
                    self.hypernet, input_dim=num_nodes*data_dim, count_bins=self.num_bins, order=self.order, bound=5.0
                )
                # AffineDiagonalPyro(input_dim=self.num_nodes*self.data_dim),
                # Spline(input_dim=self.num_nodes*self.data_dim, count_bins=self.num_bins, order="quadratic", bound=5.0),
                # AffineDiagonalPyro(input_dim=self.num_nodes*self.data_dim),
                # Spline(input_dim=self.num_nodes*self.data_dim, count_bins=self.num_bins, order="quadratic", bound=5.0),
                # AffineDiagonalPyro(input_dim=self.num_nodes*self.data_dim)
            ]
        )
        self.base_dist = distrib.Normal(
            torch.zeros(num_nodes*data_dim, device=self.device), torch.ones(
                num_nodes*data_dim, device=self.device)
        )
        # else:

        context_dict = {"X": X_history, "A": A, "embeddings": embeddings}
        flow_dist = distrib.ConditionalTransformedDistribution(
            self.base_dist, self.transform).condition(context_dict)
        return flow_dist.log_prob(X_input)

    def sample(self, 
               N_samples: int, 
               X_history: torch.Tensor,  
               W: torch.Tensor,
               embeddings: torch.Tensor):
        assert len(X_history.shape) == 4

        batch, lag, num_nodes, data_dim = X_history.shape

        base_dist = distrib.Normal(
            torch.zeros(num_nodes*data_dim, device=self.device), torch.ones(
                num_nodes*data_dim, device=self.device)
        )

        context_dict = {"X": X_history, "A": W, "embeddings": embeddings}
        flow_dist = distrib.ConditionalTransformedDistribution(
            base_dist, self.transform).condition(context_dict)
        return flow_dist.sample([N_samples, batch])
