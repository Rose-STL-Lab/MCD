import lightning.pytorch as pl
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from src.modules.adjacency_matrices.AdjMatrix import AdjMatrix


class TwoWayGraphDist(AdjMatrix, pl.LightningModule):
    """
    Sampling is performed with `torch.gumbel_softmax(..., hard=True)` to give
    binary samples and a straight-through gradient estimator.
    """

    def __init__(
        self,
        input_dim: int,
        tau_gumbel: float = 1.0
    ):
        """
        Args:
            input_dim: dimension.
            tau_gumbel: temperature used for gumbel softmax sampling.
        """
        super().__init__()
        # We only use n(n-1)/2 random samples
        # For each edge, sample either A->B, B->A or no edge
        # We convert this to a proper adjacency matrix using torch.tril_indices
        self.logits = nn.Parameter(
            torch.zeros((2, input_dim, input_dim), device=self.device), requires_grad=True
        )
        self.tau_gumbel = tau_gumbel
        self.input_dim = input_dim

    def zero_out_diagonal(self, matrix: torch.Tensor):
        # matrix: (num_nodes, num_nodes)
        N = matrix.shape[0]
        I = torch.arange(N).to(self.device)
        matrix = matrix.clone()
        matrix[I, I] = 0
        return matrix

    def get_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """
        Returns the adjacency matrix of edge probabilities.
        """
        probs = F.softmax(self.logits, dim=0)[1, ...]
        probs = self.zero_out_diagonal(probs)
        # probs = F.softmax(self.logits, dim=0)  # (3, n(n-1)/2) probabilities

        if do_round:
            return probs.round()
        return probs

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q, which is a collection of n(n-1) categoricals on 3 values.
        """
        dist = td.Categorical(logits=self.logits[1, ...] - self.logits[0, ...])
        I = torch.arange(self.input_dim)
        dist_diag = td.Categorical(
            logits=self.logits[1, I, I] - self.logits[0, I, I])
        entropies = dist.entropy()
        diag_entropy = dist_diag.entropy()
        return entropies.sum() - diag_entropy.sum()
        # return entropies.mean()

    def sample_A(self) -> torch.Tensor:
        """
        Sample an adjacency matrix from the variational distribution. It uses the gumbel_softmax trick,
        and returns hard samples (straight through gradient estimator). Adjacency returned always has
        zeros in its diagonal (no self loops).

        V1: Returns one sample to be used for the whole batch.
        """
        sample = F.gumbel_softmax(
            self.logits, tau=self.tau_gumbel, hard=True, dim=0)[1, ...]
        return self.zero_out_diagonal(sample)
