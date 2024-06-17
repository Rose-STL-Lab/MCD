import lightning.pytorch as pl
from torch import nn
import torch
from src.modules.MultiEmbedding import MultiEmbedding
from src.utils.torch_utils import generate_fully_connected


class MultiCausalDecoder(pl.LightningModule):

    def __init__(self,
                 data_dim: int,
                 lag: int,
                 num_nodes: int,
                 num_graphs: int,
                 embedding_dim: int = None,
                 skip_connection: bool = False,
                 linear: bool = False,
                 dropout_p: float = 0.0
                 ):

        super().__init__()

        if embedding_dim is not None:
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = num_nodes * data_dim

        self.data_dim = data_dim
        self.lag = lag
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.dropout_p = dropout_p
        self.linear = linear

        input_dim = 2*self.embedding_dim

        if not self.linear:
            self.nn_size = max(4 * num_nodes, self.embedding_dim, 64)

            self.f = generate_fully_connected(
                input_dim=input_dim,
                output_dim=num_nodes*data_dim,  # potentially num_nodes
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

            self.cd_embeddings = MultiEmbedding(num_nodes=self.num_nodes,
                                                lag=self.lag,
                                                num_graphs=self.num_graphs,
                                                embedding_dim=self.embedding_dim)

        else:
            self.w = nn.Parameter(
                torch.randn(self.num_graphs, self.lag+1, self.num_nodes, self.num_nodes, device=self.device)*0.5, requires_grad=True
            )

    def forward(self, X_input: torch.Tensor, A: torch.Tensor):
        """
        Args:
            X_input: input data of shape (batch, lag+1, num_nodes, data_dim)
            A: adjacency matrix of shape (num_graphs, lag+1, num_nodes, num_nodes)
        """

        assert len(X_input.shape) == 4

        batch, L, num_nodes, data_dim = X_input.shape
        lag = L-1

        if not self.linear:
            E = self.cd_embeddings.get_embeddings()

            # reshape X to the correct shape
            A = A.unsqueeze(0).expand((batch, -1, -1, -1, -1))
            E = E.unsqueeze(0).expand((batch, -1, -1, -1, -1))
            X_input = X_input.unsqueeze(1).expand(
                (-1, self.num_graphs, -1, -1, -1))

            # ensure we have the correct shape
            assert (A.shape[0] == batch and A.shape[1] == self.num_graphs and
                    A.shape[2] == lag + 1 and A.shape[3] == num_nodes and
                    A.shape[4] == num_nodes)
            assert (E.shape[0] == batch and E.shape[1] == self.num_graphs
                    and E.shape[2] == lag+1 and E.shape[3] == num_nodes
                    and E.shape[4] == self.embedding_dim)

            X_in = torch.cat((X_input, E), dim=-1)
            # X_in: (batch, lag+1, num_nodes, embedding_dim+data_dim)
            X_enc = self.g(X_in)
            A_temp = A.flip([2])
            # get the parents of X
            X_sum = torch.einsum("bnlij,bnlio->bnjo", A_temp, X_enc)

            X_sum = torch.cat([X_sum, E[:, :, 0, :, :]], dim=-1)
            # (batch, num_graphs, num_nodes, embedding_dim)
            # pass through f network to get the predictions
            group_mask = torch.eye(num_nodes*data_dim).to(self.device)

            # (batch, num_graphs, num_nodes, data_dim)
            return torch.sum(self.f(X_sum)*group_mask, dim=-1).unsqueeze(-1)

        return torch.einsum("klij,blio->bkjo", (self.w * A).flip([1]), X_input)
        # return self.f(X_sum)
