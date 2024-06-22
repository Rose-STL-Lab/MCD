from typing import Any
import numpy as np
import torch

# import tigramite for pcmci
import networkx as nx
import pandas as pd

from src.baselines.BaselineTrainer import BaselineTrainer
from src.modules.dynotears.dynotears import from_pandas_dynamic
from src.utils.data_utils.data_format_utils import to_time_aggregated_graph_np, to_time_aggregated_scores_np, zero_out_diag_np

class DYNOTEARSTrainer(BaselineTrainer):

    def __init__(self,
                 full_dataset: np.array,
                 adj_matrices: np.array,
                 data_dim: int,
                 num_nodes: int,
                 lag: int,
                 num_workers: int = 16,
                 aggregated_graph: bool = False,
                 single_graph: bool = True,
                 max_iter: int = 1000,
                 lambda_w: float = 0.1,
                 lambda_a: float = 0.1,
                 w_threshold: float = 0.4,
                 h_tol: float = 1e-8,
                 group_by_graph: bool = True,
                 ignore_self_connections: bool = False
                 ):
        self.group_by_graph = group_by_graph
        self.ignore_self_connections = ignore_self_connections
        if self.group_by_graph:
            self.single_graph = True
            print("DYNOTEARS: Group by graph option set. Overriding single graph flag to True...")
        else:
            self.single_graph = single_graph
        super().__init__(full_dataset=full_dataset,
                         adj_matrices=adj_matrices,
                         data_dim=data_dim,
                         lag=lag,
                         num_nodes=num_nodes,
                         num_workers=num_workers,
                         aggregated_graph=aggregated_graph)

        self.max_iter = max_iter
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.w_threshold = w_threshold
        self.h_tol = h_tol

        if self.single_graph:
            self.batch_size = full_dataset.shape[0] # we want the full dataset
        else:
            self.batch_size = 1

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        X, adj_matrix, graph_index = batch

        batch_size, timesteps, num_nodes, _ = X.shape
        assert num_nodes == self.num_nodes
        X = X.view(batch_size, timesteps, -1)

        X, adj_matrix, graph_index = X.cpu().numpy(), adj_matrix.cpu().numpy(), graph_index.cpu().numpy()

        X_list = []

        graphs = np.zeros((batch_size, self.lag+1, num_nodes, num_nodes))
        scores = np.zeros((batch_size, self.lag+1, num_nodes, num_nodes))
        if self.group_by_graph:
            n_unique_matrices = np.max(graph_index)+1
        else:
            graph_index = np.zeros((batch_size))
            n_unique_matrices = 1

        for i in range(n_unique_matrices):

            n_samples = np.sum(graph_index == i)
            for x in X[graph_index == i]:
                X_list.append(pd.DataFrame(x))
            learner = from_pandas_dynamic(
                X_list,
                p=self.lag,
                max_iter=self.max_iter,
                lambda_w=self.lambda_w,
                lambda_a=self.lambda_a,
                w_threshold=self.w_threshold,
                h_tol=self.h_tol)

            adj_static = nx.to_numpy_array(learner)
            temporal_adj_list = []
            for l in range(self.lag + 1):
                cur_adj = adj_static[l:: self.lag + 1, 0:: self.lag + 1]
                temporal_adj_list.append(cur_adj)

            # [lag+1, num_nodes, num_nodes]
            score = np.stack(temporal_adj_list, axis=0)
            # scores = np.hstack(temporal_adj_list)
            temporal_adj = [(score != 0).astype(int) for _ in range(n_samples)]
            score = [np.abs(score) for _ in range(n_samples)]
            graphs[i == graph_index] = np.array(temporal_adj)
            scores[i == graph_index] = np.array(score)
        if self.aggregated_graph:
            graphs = to_time_aggregated_graph_np(graphs)
            scores = to_time_aggregated_scores_np(scores)
            if self.ignore_self_connections:
                graphs = zero_out_diag_np(graphs)
                scores = zero_out_diag_np(scores)
        return torch.Tensor(graphs), torch.abs(torch.Tensor(scores)), torch.Tensor(adj_matrix)
    