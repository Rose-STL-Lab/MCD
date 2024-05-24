from typing import Any
from src.baselines.BaselineTrainer import BaselineTrainer
import numpy as np
from src.utils.data_utils.data_format_utils import to_time_aggregated_graph_np, zero_out_diag_np, zero_out_diag_torch
# import tigramite for pcmci
import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn
from copy import deepcopy
import torch
from src.utils.causality_utils import *

"""
Large parts adapted from https://github.com/microsoft/causica
"""

class PCMCITrainer(BaselineTrainer):

    def __init__(self,
                 full_dataset: np.array,
                 adj_matrices: np.array,
                 data_dim: int,
                 num_nodes: int,
                 lag: int,
                 num_workers: int = 16,
                 aggregated_graph: bool = False,
                 ci_test: str = 'ParCorr', # options are ParCorr, CMIknn, GPDCtorch
                 single_graph: bool = False,
                 pcmci_plus: bool = True,
                 pc_alpha: float = 0.01,
                 group_by_graph: bool = False,
                 ignore_self_connections: bool = False
                 ):
        
        self.group_by_graph = group_by_graph
        self.ignore_self_connections = ignore_self_connections
        if self.group_by_graph:
            print("PCMCI: Group by graph option set. Overriding single graph flag to True...")
            self.single_graph = True
        else:
            self.single_graph = single_graph

        super().__init__(full_dataset=full_dataset,
                         adj_matrices=adj_matrices,
                         data_dim=data_dim,
                         lag=lag,
                         num_nodes=num_nodes,
                         num_workers=num_workers,
                         aggregated_graph=aggregated_graph)

        if ci_test == 'ParCorr':
            self.ci_test = ParCorr(significance='analytic')
        elif ci_test == 'CMIknn':
            self.ci_test = CMIknn()

        # self.single_graph = single_graph
        self.pcmci_plus = pcmci_plus
        self.pc_alpha = pc_alpha

        if self.single_graph:
            self.batch_size = full_dataset.shape[0] # we want the full dataset
            self.analysis_mode = 'multiple'
        else:
            self.batch_size = 1
            self.analysis_mode = 'single'

    def _process_adj_matrix(self, adj_matrix: np.ndarray) -> np.ndarray:
        """
        Borrowed from microsoft/causica

        This will process the raw output adj graphs from pcmci_plus. The raw output can contain 3 types of edges:
            (1) "-->" or "<--". This indicates the directed edges, and they should appear symmetrically in the matrix.
            (2) "o-o": This indicates the bi-directed edges, also appears symmetrically.
            Note: for lagged matrix, it can only contain "-->".
            (3) "x-x": this means the edge direction is un-decided due to conflicting orientation rules. We ignores
                the edges in this case.
        Args:
            inst_matrix: the input raw inst matrix with shape [num_nodes, num_nodes, lag+1]
        Returns:
            inst_adj_matrix: np.ndarray, an inst adj matrix with shape [lag+1, num_nodes, num_nodes]
        """
        assert adj_matrix.ndim == 3

        adj_matrix = deepcopy(adj_matrix)
        # shape [lag+1, num_nodes, num_nodes]
        adj_matrix = np.moveaxis(adj_matrix, -1, 0)
        adj_matrix[adj_matrix == ""] = 0
        adj_matrix[adj_matrix == "<--"] = 0
        adj_matrix[adj_matrix == "-->"] = 1
        adj_matrix[adj_matrix == "o-o"] = 1
        adj_matrix[adj_matrix == "x-x"] = 0

        return adj_matrix.astype(int)

    def _run_pcmci(self, pcmci, tau_max, pc_alpha):
        if self.pcmci_plus:
            return pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)
        else:
            return pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)

    def _process_cpdag(self, adj_matrix: np.ndarray):
        """
        This will process the inst cpdag (i.e. adj_matrix[0, ...]) according to the mec_mode. It supports "enumerate" and "truth"
        Args:
            adj_matrix: np.ndarray, a temporal adj matrix with shape [lag+1, num_nodes, num_nodes] where the inst part can be a cpdag.

        Returns:
            adj_matrix: np.ndarray with shape [num_possible_dags, lag+1, num_nodes, num_nodes]
        """
        
        lag_plus, num_nodes = adj_matrix.shape[0], adj_matrix.shape[1]
        static_temporal_graph = convert_temporal_to_static_adjacency_matrix(
            adj_matrix, conversion_type="auto_regressive"
        )  # shape[(lag+1) *nodes, (lag+1)*nodes]
        all_static_temp_dags = cpdag2dags(
            static_temporal_graph, samples=3000
        )  # [all_possible_dags, (lag+1)*num_nodes, (lag+1)*num_nodes]
        # convert back to temporal adj matrix.
        temp_adj_list = np.split(
            all_static_temp_dags[..., :, (lag_plus - 1) * num_nodes :], lag_plus, axis=1
        )  # list with length lag+1, each with shape [all_possible_dags, num_nodes, num_nodes]
        proc_adj_matrix = np.stack(
            list(reversed(temp_adj_list)), axis=-3
        )  # shape [all_possible_dags, lag+1, num_nodes, num_nodes]

        return proc_adj_matrix


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        X, adj_matrix, graph_index = batch

        batch_size, timesteps, num_nodes, data_dim = X.shape
        assert num_nodes == self.num_nodes
        
        X = X.view(batch_size, timesteps, -1)
        X, adj_matrix, graph_index = X.numpy(), adj_matrix.numpy(), graph_index.numpy()
        
        graphs = [] #np.zeros((batch_size, self.lag+1, num_nodes, num_nodes))
        new_adj_matrix = []
        if self.group_by_graph:
            n_unique_matrices = np.max(graph_index)+1
        else:
            graph_index = np.zeros((batch_size))
            n_unique_matrices = 1

        unique_matrices = np.unique(adj_matrix, axis=0)
        for i in range(n_unique_matrices):
            print(f"{i}/{n_unique_matrices}")
            n_samples = np.sum(graph_index == i)
            dataframe = pp.DataFrame(X[graph_index==i], analysis_mode=self.analysis_mode)
            pcmci = PCMCI(
                dataframe=dataframe,
                cond_ind_test=self.ci_test,
                verbosity=0)

            results = self._run_pcmci(pcmci, self.lag, self.pc_alpha)

            graph = self._process_adj_matrix(results["graph"])
            
            graph = self._process_cpdag(graph)

            num_possible_dags = graph.shape[0]

            new_adj_matrix.append(np.repeat(adj_matrix[graph_index==i][0][np.newaxis, ...], n_samples*num_possible_dags, axis=0))
            graphs.append(np.repeat(graph, n_samples, axis=0))

        graphs = np.concatenate(graphs, axis=0)
        new_adj_matrix = np.concatenate(new_adj_matrix, axis=0)
        if self.aggregated_graph:
            graphs = to_time_aggregated_graph_np(graphs)
            if self.ignore_self_connections:
                graphs = zero_out_diag_np(graphs)

        return torch.Tensor(graphs), torch.Tensor(graphs), torch.Tensor(new_adj_matrix)