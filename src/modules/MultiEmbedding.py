import lightning.pytorch as pl
import torch.nn as nn
import torch

class MultiEmbedding(pl.LightningModule):
    
    def __init__(
        self,
        num_nodes: int,
        lag: int,
        num_graphs: int,
        embedding_dim: int
    ):

        super().__init__()
        self.lag = lag
        # Assertion lag > 0
        assert lag > 0
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.embedding_dim = embedding_dim

        self.lag_embeddings = nn.Parameter((
                torch.randn(self.num_graphs, self.lag, self.num_nodes,
                            self.embedding_dim, device=self.device) * 0.01
            ), requires_grad=True) 

        self.inst_embeddings = nn.Parameter((
                torch.randn(self.num_graphs, 1, self.num_nodes,
                            self.embedding_dim, device=self.device) * 0.01
            ), requires_grad=True) 
    
    def turn_off_inst_grad(self):
        self.inst_embeddings.requires_grad_(False)

    def turn_on_inst_grad(self):
        self.inst_embeddings.requires_grad_(True)

    def get_embeddings(self):
        return torch.cat((self.inst_embeddings, self.lag_embeddings), dim=1)
        
    