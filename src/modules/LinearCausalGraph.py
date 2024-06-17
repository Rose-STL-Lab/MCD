
import lightning.pytorch as pl
import torch
from torch import nn


class LinearCausalGraph(pl.LightningModule):

    def __init__(
        self,
        lag: int,
        input_dim: int
    ):
        """
        Args:
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__()

        self.lag = lag
        self.w = nn.Parameter(
            torch.zeros(self.lag+1, input_dim, input_dim, device=self.device), requires_grad=True
        )
        self.I = torch.arange(input_dim)
        self.mask = torch.ones((self.lag+1, input_dim, input_dim))
        self.mask[0, self.I, self.I] = 0
        self.input_dim = input_dim

    def get_w(self) -> torch.Tensor:
        """
        Returns the matrix. Ensures that the instantaneous matrix has zero in the diagonals
        """
        return self.w * self.mask.to(self.device)
