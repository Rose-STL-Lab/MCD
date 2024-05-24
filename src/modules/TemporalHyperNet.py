import lightning.pytorch as pl
import torch.nn as nn
import torch
from typing import Dict, List, Optional, Tuple, Type
import math
from src.utils.torch_utils import generate_fully_connected

class TemporalHyperNet(pl.LightningModule):

    def __init__(self,
                 order: str,
                 lag: int,
                 data_dim: int,
                 num_nodes: int,
                 embedding_dim: int = None,
                 skip_connection: bool = False,
                 num_bins: int = 8):
        super().__init__()

        if embedding_dim is None:
            embedding_dim = num_nodes * data_dim

        self.embedding_dim = embedding_dim
        self.data_dim = data_dim
        self.lag = lag
        self.order = order
        self.num_bins = num_bins
        self.num_nodes = num_nodes
        
        if self.order == "quadratic":
            self.param_dim = [
                self.num_bins,
                self.num_bins,
                (self.num_bins - 1),
            ]  # this is for quadratic order conditional spline flow
        elif self.order == "linear":
            self.param_dim = [
                self.num_bins,
                self.num_bins,
                (self.num_bins - 1),
                self.num_bins,
            ]  # this is for linear order conditional spline flow
        
        self.total_param = sum(self.param_dim)
        input_dim = 2*self.embedding_dim
        self.nn_size = max(4 * num_nodes, self.embedding_dim, 64)

        self.f = generate_fully_connected(
            input_dim=input_dim,
            output_dim=self.total_param, #potentially num_nodes
            hidden_dims=[self.nn_size, self.nn_size],
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=nn.LayerNorm,
            res_connection=skip_connection,
        )

        self.g = generate_fully_connected(
            input_dim=self.embedding_dim+self.data_dim,
            output_dim=self.embedding_dim,
            hidden_dims=[self.nn_size, self.nn_size],
            non_linearity=nn.LeakyReLU,
            activation=nn.Identity,
            device=self.device,
            normalization=nn.LayerNorm,
            res_connection=skip_connection,
        )

        self.embeddings = nn.Parameter((
            torch.randn(self.lag + 1, self.num_nodes,
                    self.embedding_dim, device=self.device) * 0.01
        ), requires_grad=True)  # shape (lag+1, num_nodes, embedding_dim)

            
    def forward(self, X: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            X: A dict consisting of two keys, "A" is the adjacency matrix with shape [batch, lag+1, num_nodes, num_nodes]
            and "X" is the history data with shape (batch, lag, num_nodes, data_dim).

        Returns:
            A tuple of parameters with shape [N_batch, num_cts_node*param_dim_each].
                The length of tuple is len(self.param_dim),
        """

        # assert "A" in X and "X" in X and len(
        # X) == 2, "The key for input can only contain two keys, 'A', 'X'."

        A = X["A"]
        X_in = X["X"]
        embeddings = X["embeddings"]
        batch, lag, num_nodes, data_dim = X_in.shape

        # ensure we have the correct shape
        assert (A.shape[0] == batch and A.shape[1] == lag +
                1 and A.shape[2] == num_nodes and A.shape[3] == num_nodes)

        if embeddings == None:
            E = self.embeddings.expand(
                X_in.shape[0], -1, -1, -1
            )
        else:
            E = embeddings

        # shape [batch_size, lag, num_nodes, embedding_size]
        E_lag = E[..., 1:, :, :]

        X_in = torch.cat((X_in, E_lag), dim=-1)
        # X_in: (batch, lag, num_nodes, embedding_dim + data_dim)

        X_enc = self.g(X_in)  # (batch, lag, num_nodes, embedding_dim)

        # get the parents of X
        # (batch, num_nodes, embedding_dim)
        A_temp = A[:, 1:].flip([1])

        X_sum = torch.einsum("blij,blio->bjo", A_temp, X_enc) #/ num_nodes

        X_sum = torch.cat((X_sum, E[..., 0, :, :]), dim=-1)

        # pass through f network to get the parameters
        params = self.f(X_sum)  # (batch, num_nodes, total_params)

        param_list = torch.split(params, self.param_dim, dim=-1)
        # a list of tensor with shape [batch, num_nodes*each_param]
        return tuple(
            param.reshape([-1, num_nodes * param.shape[-1]]) for param in param_list)
