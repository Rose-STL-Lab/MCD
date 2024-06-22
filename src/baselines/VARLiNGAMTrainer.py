from typing import Any
import numpy as np

import lingam
import torch

from src.utils.data_utils.data_format_utils import to_time_aggregated_graph_np
from src.baselines.BaselineTrainer import BaselineTrainer

class VARLiNGAMTrainer(BaselineTrainer):

    def __init__(self,
                 full_dataset: np.array,
                 adj_matrices: np.array,
                 data_dim: int,
                 num_nodes: int,
                 lag: int,
                 num_workers: int = 16,
                 aggregated_graph: bool = False
                 ):
        super().__init__(full_dataset=full_dataset,
                         adj_matrices=adj_matrices,
                         data_dim=data_dim,
                         lag=lag,
                         num_nodes=num_nodes,
                         num_workers=num_workers,
                         aggregated_graph=aggregated_graph)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        X, adj_matrix, _ = batch

        batch, timesteps, num_nodes, _ = X.shape
        X = X.view(batch, timesteps, -1)

        assert num_nodes == self.num_nodes
        assert batch == 1, "VARLiNGAM needs batch size 1"

        model_pruned = lingam.VARLiNGAM(lags=self.lag, prune=True)
        model_pruned.fit(X[0])
        graph = np.transpose(np.abs(model_pruned.adjacency_matrices_) > 0, axes=[0, 2, 1])
        if graph.shape[0] != (self.lag+1):
            while graph.shape[0] != (self.lag+1):
                graph = np.concatenate((graph, np.zeros((1, num_nodes, num_nodes) )), axis=0)
        graphs = [graph]
        if self.aggregated_graph:
            graphs = to_time_aggregated_graph_np(graphs)
        print(graphs)
        print(adj_matrix)
        return torch.Tensor(graphs), torch.Tensor(graphs), torch.Tensor(adj_matrix)