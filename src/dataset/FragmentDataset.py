"""
Terminology:

A fragment refers to the pair of tensors X_history and x_current, where X_history represents
the lag information (X(t-lag) to X(t-1)) and x_current represents the current information X(t).
Note that which sample a fragment comes from is irrelevant, since all we are concerned about is the
causal graph which generated the fragment.
"""

from torch.utils.data import Dataset
import torch

from src.utils.data_utils.data_format_utils import convert_data_to_timelagged, convert_adj_to_timelagged


class FragmentDataset(Dataset):
    def __init__(self,
                 X,
                 adj_matrix,
                 lag,
                 return_graph_indices=True,
                 aggregated_graph=False):
        """
        X: np.array of shape (n_samples, timesteps, num_nodes, data_dim)
        adj_matrix: np.array of shape (n_samples, lag+1, num_nodes, num_nodes)
        """
        self.lag = lag
        self.aggregated_graph = aggregated_graph
        self.return_graph_indices = return_graph_indices
        # preprocess data
        self.X_history, self.x_current, self.X_indices = convert_data_to_timelagged(
            X, lag=lag)
        if self.return_graph_indices:
            self.adj_matrix, self.graph_indices = convert_adj_to_timelagged(
                adj_matrix,
                lag=lag,
                n_fragments=self.X_history.shape[0],
                aggregated_graph=self.aggregated_graph,
                return_indices=True)
        else:
            self.adj_matrix = convert_adj_to_timelagged(
                adj_matrix,
                lag=lag,
                n_fragments=self.X_history.shape[0],
                aggregated_graph=self.aggregated_graph,
                return_indices=False)

        self.X_history, self.x_current, self.adj_matrix, self.X_indices = \
            torch.Tensor(self.X_history), torch.Tensor(self.x_current), torch.Tensor(
                self.adj_matrix), torch.Tensor(self.X_indices)
        if self.return_graph_indices:
            self.graph_indices = torch.Tensor(self.graph_indices)

        self.num_fragments = self.X_history.shape[0]

    def __len__(self):
        return self.num_fragments

    def __getitem__(self, index):
        if self.return_graph_indices:
            return self.X_history[index], self.x_current[index], self.adj_matrix[index], self.X_indices[index].long(), \
                self.graph_indices[index].long()
        return self.X_history[index], self.x_current[index], self.adj_matrix[index], self.X_indices[index].long()
