import lightning.pytorch as pl
from torch import nn
import torch
from src.utils.torch_utils import generate_fully_connected


class CausalDecoder(pl.LightningModule):

    def __init__(self,
                 data_dim: int,
                 lag: int,
                 num_nodes: int,
                 embedding_dim: int = None,
                 skip_connection: bool = False,
                 linear: bool = False
                 ):
        super().__init__()

        if embedding_dim is None:
            embedding_dim = num_nodes * data_dim

        self.embedding_dim = embedding_dim
        self.data_dim = data_dim
        self.lag = lag
        self.num_nodes = num_nodes
        self.linear = linear

        if not self.linear:
            self.embeddings = nn.Parameter((
                torch.randn(self.lag + 1, self.num_nodes,
                            self.embedding_dim, device=self.device) * 0.01
            ), requires_grad=True)  # shape (lag+1, num_nodes, embedding_dim)

            input_dim = 2*self.embedding_dim
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

        else:
            self.w = nn.Parameter(
                torch.randn(self.lag+1, self.num_nodes, self.num_nodes, device=self.device)*0.5, requires_grad=True
            )

    def forward(self, X_input: torch.Tensor, A: torch.Tensor, embeddings: torch.Tensor = None):
        """
        Args:
            X_input: input data of shape (batch, lag+1, num_nodes, data_dim)
            A: adjacency matrix of shape (batch, (lag+1), num_nodes, num_nodes)
        """

        assert len(X_input.shape) == 4

        batch, L, num_nodes, data_dim = X_input.shape
        lag = L-1

        if not self.linear:
            # ensure we have the correct shape
            assert (A.shape[0] == batch and A.shape[1] == lag+1 and A.shape[2] ==
                    num_nodes and A.shape[2] == num_nodes)

            if embeddings is None:
                E = self.embeddings.expand(
                    X_input.shape[0], -1, -1, -1
                )
            else:
                E = embeddings

            X_in = torch.cat((X_input, E), dim=-1)
            # X_in: (batch, lag+1, num_nodes, embedding_dim+data_dim)
            X_enc = self.g(X_in)

            A_temp = A.flip([1])

            # get the parents of X
            X_sum = torch.einsum("blij,blio->bjo", A_temp,
                                 X_enc)  # / num_nodes

            X_sum = torch.cat([X_sum, E[:, 0, :, :]], dim=-1)
            # (batch, num_nodes, embedding_dim)
            # pass through f network to get the predictions

            group_mask = torch.eye(num_nodes*data_dim).to(self.device)
            # (batch, num_nodes, data_dim)
            return torch.sum(self.f(X_sum)*group_mask, dim=-1).unsqueeze(-1)
        return torch.einsum("lij,blio->bjo", (self.w * A[0]).flip([0]), X_input)
