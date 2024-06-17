from typing import List, Optional

import lightning.pytorch as pl
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from src.modules.adjacency_matrices.AdjMatrix import AdjMatrix


class MultiTemporalAdjacencyMatrix(pl.LightningModule, AdjMatrix):
    def __init__(
        self,
        num_nodes: int,
        lag: int,
        num_graphs: int,
        tau_gumbel: float = 1.0,
        threeway: bool = True,
        init_logits: Optional[List[float]] = None,
        disable_inst: bool = False
    ):
        super().__init__()
        self.lag = lag
        self.tau_gumbel = tau_gumbel
        self.threeway = threeway
        self.num_nodes = num_nodes
        # Assertion lag > 0
        assert lag > 0
        self.num_graphs = num_graphs
        self.disable_inst = disable_inst

        if self.threeway:
            self.logits_inst = nn.Parameter(
                torch.zeros((3, self.num_graphs, (num_nodes * (num_nodes - 1)) // 2),
                            device=self.device),
                requires_grad=True
            )
            self.lower_idxs = torch.unbind(
                torch.tril_indices(self.num_nodes, self.num_nodes,
                                   offset=-1, device=self.device), 0
            )
        else:
            self.logits_inst = nn.Parameter(
                torch.zeros((2, self.num_graphs, num_nodes, num_nodes),
                            device=self.device),
                requires_grad=True
            )

        self.logits_lag = nn.Parameter(torch.zeros((2, self.num_graphs, lag, num_nodes, num_nodes),
                                                   device=self.device),
                                       requires_grad=True)
        self.init_logits = init_logits
        # Set the init_logits if not None
        if self.init_logits is not None:
            if self.threeway:
                self.logits_inst.data[2, :, ...] = self.init_logits[0]
            else:
                self.logits_inst.data[1, :, ...] = self.init_logits[0]
            self.logits_lag.data[0, :, ...] = self.init_logits[1]

    def zero_out_diagonal(self, matrix: torch.Tensor):
        # matrix: (num_graphs, num_nodes, num_nodes)
        N = matrix.shape[1]
        I = torch.arange(N).to(self.device)
        matrix = matrix.clone()
        matrix[:, I, I] = 0
        return matrix

    def _triangular_vec_to_matrix(self, vec):
        """
        Given an array of shape (k, N, n(n-1)/2) where k in {2, 3}, creates a matrix of shape
        (N, n, n) where the lower triangular is filled from vec[0, :] and the upper
        triangular is filled from vec[1, :].
        """
        N = vec.shape[1]
        output = torch.zeros(
            (N, self.num_nodes, self.num_nodes), device=self.device)
        output[:, self.lower_idxs[0], self.lower_idxs[1]] = vec[0, :, ...]
        output[:, self.lower_idxs[1], self.lower_idxs[0]] = vec[1, :, ...]
        return output

    def get_adj_matrix(self, do_round: bool = False) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        probs = torch.zeros((self.num_graphs, self.lag + 1, self.num_nodes, self.num_nodes),
                            device=self.device)

        if not self.disable_inst:
            inst_probs = F.softmax(self.logits_inst, dim=0)
            if self.threeway:
                # (3, n(n-1)/2) probabilities
                inst_probs = self._triangular_vec_to_matrix(inst_probs)
            else:
                inst_probs = self.zero_out_diagonal(inst_probs[1, ...])

            # Generate simultaneous adj matrix
            # shape (input_dim, input_dim)
            probs[:, 0, ...] = inst_probs

        # Generate lagged adj matrix
        # shape (lag, input_dim, input_dim)
        probs[:, 1:, ...] = F.softmax(self.logits_lag, dim=0)[1, ...]
        if do_round:
            return probs.round()

        return probs

    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q. In this case 0.
        """

        if not self.disable_inst:
            if self.threeway:
                dist = td.Categorical(
                    logits=self.logits_inst[:, :].transpose(0, -1))
                entropies_inst = dist.entropy().sum()
            else:
                dist = td.Categorical(
                    logits=self.logits_inst[1, ...] - self.logits_inst[0, ...])
                I = torch.arange(self.num_nodes)
                dist_diag = td.Categorical(
                    logits=self.logits_inst[1, :, I, I] - self.logits_inst[0, :, I, I])
                entropies = dist.entropy()
                diag_entropy = dist_diag.entropy()
                entropies_inst = entropies.sum() - diag_entropy.sum()
        else:
            entropies_inst = 0

        dist_lag = td.Independent(td.Bernoulli(
            logits=self.logits_lag[1, :] - self.logits_lag[0, :]), 3)
        entropies_lag = dist_lag.entropy().sum()

        return entropies_lag + entropies_inst

    def sample_A(self) -> torch.Tensor:
        """
        This samples the adjacency matrix from the variational distribution. This uses the gumbel softmax trick and returns
        hard samples. This can be done by (1) sample instantaneous adj matrix using self.logits, (2) sample lagged adj matrix using self.logits_lag.
        """
        # Create adj matrix to avoid concatenation
        adj_sample = torch.zeros(
            (self.num_graphs, self.lag + 1, self.num_nodes, self.num_nodes), device=self.device
        )  # shape ( lag+1, input_dim, input_dim)

        if not self.disable_inst:
            if self.threeway:
                # Sample instantaneous adj matrix
                adj_sample[:, 0, ...] = self._triangular_vec_to_matrix(
                    F.gumbel_softmax(self.logits_inst,
                                     tau=self.tau_gumbel,
                                     hard=True,
                                     dim=0)
                )  # shape (N, input_dim, input_dim)
            else:
                sample = F.gumbel_softmax(
                    self.logits_inst, tau=self.tau_gumbel, hard=True, dim=0)[1, ...]
                adj_sample[:, 0, ...] = self.zero_out_diagonal(sample)

        # Sample lagged adj matrix
        # shape (N, lag, input_dim, input_dim)
        adj_sample[:, 1:, ...] = F.gumbel_softmax(self.logits_lag,
                                                  tau=self.tau_gumbel,
                                                  hard=True,
                                                  dim=0)[1, ...]
        return adj_sample

    def turn_off_inst_grad(self):
        self.logits_inst.requires_grad_(False)

    def turn_on_inst_grad(self):
        self.logits_inst.requires_grad_(True)
