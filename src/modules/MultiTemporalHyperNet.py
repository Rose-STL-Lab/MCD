from typing import Dict, Tuple
import lightning.pytorch as pl
from torch import nn
import torch
from src.modules.MultiEmbedding import MultiEmbedding
from src.utils.torch_utils import generate_fully_connected


class MultiTemporalHyperNet(pl.LightningModule):

    def __init__(self,
                 order: str,
                 lag: int,
                 data_dim: int,
                 num_nodes: int,
                 num_graphs: int,
                 embedding_dim: int = None,
                 skip_connection: bool = False,
                 num_bins: int = 8,
                 dropout_p: float = 0.0
                 ):

        super().__init__()

        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = num_nodes * data_dim

        self.data_dim = data_dim
        self.lag = lag
        self.order = order
        self.num_bins = num_bins
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.dropout_p = dropout_p

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
            output_dim=self.total_param,  # potentially num_nodes
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

        self.th_embeddings = MultiEmbedding(num_nodes=self.num_nodes,
                                            lag=self.lag,
                                            num_graphs=self.num_graphs,
                                            embedding_dim=self.embedding_dim)

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

        E = self.th_embeddings.get_embeddings()
        # X["embeddings"]

        batch, lag, num_nodes, _ = X_in.shape

        # reshape X to the correct shape
        A = A.unsqueeze(0).expand((batch, -1, -1, -1, -1))
        E = E.unsqueeze(0).expand((batch, -1, -1, -1, -1))
        X_in = X_in.unsqueeze(1).expand((-1, self.num_graphs, -1, -1, -1))

        # ensure we have the correct shape
        assert (A.shape[0] == batch and A.shape[1] == self.num_graphs and
                A.shape[2] == lag + 1 and A.shape[3] == num_nodes and
                A.shape[4] == num_nodes)
        assert (E.shape[0] == batch and E.shape[1] == self.num_graphs
                and E.shape[2] == lag+1 and E.shape[3] == num_nodes
                and E.shape[4] == self.embedding_dim)

        # shape [batch_size, num_graphs, lag, num_nodes, embedding_size]
        E_lag = E[:, :, 1:, :, :]

        # reshape X to the correct shape
        X_in = torch.cat((X_in, E_lag), dim=-1)

        # X_in: (batch, num_graphs, lag, num_nodes, embedding_dim + data_dim)

        X_enc = self.g(X_in)  # (batch, lag, num_nodes, embedding_dim)

        # get the parents of X
        # (batch, num_graphs, num_nodes, embedding_dim)
        A_temp = A[:, :, 1:].flip([2])

        X_sum = torch.einsum("bnlij,bnlio->bnjo", A_temp, X_enc)  # / num_nodes

        X_sum = torch.cat((X_sum, E[:, :, 0, :, :]), dim=-1)

        # pass through f network to get the parameters
        params = self.f(X_sum)  # (batch, num_graphs, num_nodes, total_params)

        # pass multiple graph parameters along batch dimension
        params = params.reshape(batch*self.num_graphs, num_nodes, -1)

        param_list = torch.split(params, self.param_dim, dim=-1)
        # a list of tensor with shape [batch*num_graphs, num_nodes*each_param]
        return tuple(
            param.reshape([-1, num_nodes * param.shape[-1]]) for param in param_list)
