
import lightning.pytorch as pl
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from src.modules.adjacency_matrices.AdjMatrix import AdjMatrix

class ThreeWayGraphDist(AdjMatrix, pl.LightningModule):
    """
    An alternative variational distribution for graph edges. For each pair of nodes x_i and x_j
    where i < j, we sample a three way categorical C_ij. If C_ij = 0, we sample the edge
    x_i -> x_j, if C_ij = 1, we sample the edge x_j -> x_i, and if C_ij = 2, there is no
    edge between these nodes. This variational distribution is faster to use than ENCO
    because it avoids any calls to `torch.stack`.

    Sampling is performed with `torch.gumbel_softmax(..., hard=True)` to give
    binary samples and a straight-through gradient estimator.
    """

    def __init__(
        self,
        input_dim: int,
        tau_gumbel: float = 1.0,
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
            torch.zeros(3, (input_dim * (input_dim - 1)) // 2, device=self.device), requires_grad=True
        )
        self.tau_gumbel = tau_gumbel
        self.input_dim = input_dim
        self.lower_idxs = torch.unbind(
            torch.tril_indices(self.input_dim, self.input_dim, offset=-1, device=self.device), 0
        )

    def _triangular_vec_to_matrix(self, vec):
        """
        Given an array of shape (k, n(n-1)/2) where k in {2, 3}, creates a matrix of shape
        (n, n) where the lower triangular is filled from vec[0, :] and the upper
        triangular is filled from vec[1, :].
        """
        output = torch.zeros((self.input_dim, self.input_dim), device=self.device)
        output[self.lower_idxs[0], self.lower_idxs[1]] = vec[0, ...]
        output[self.lower_idxs[1], self.lower_idxs[0]] = vec[1, ...]
        return output

    def get_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """
        Returns the adjacency matrix of edge probabilities.
        """
        probs = F.softmax(self.logits, dim=0)  # (3, n(n-1)/2) probabilities
        out_probs = self._triangular_vec_to_matrix(probs)
        if do_round:
            return out_probs.round()
        else:
            return out_probs

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q, which is a collection of n(n-1) categoricals on 3 values.
        """
        dist = td.Categorical(logits=self.logits.transpose(0, -1))
        entropies = dist.entropy()
        return entropies.sum()
        # return entropies.mean()

    def sample_A(self) -> torch.Tensor:
        """
        Sample an adjacency matrix from the variational distribution. It uses the gumbel_softmax trick,
        and returns hard samples (straight through gradient estimator). Adjacency returned always has
        zeros in its diagonal (no self loops).

        V1: Returns one sample to be used for the whole batch.
        """
        sample = F.gumbel_softmax(self.logits, tau=self.tau_gumbel, hard=True, dim=0)  # (3, n(n-1)/2) binary
        return self._triangular_vec_to_matrix(sample)

