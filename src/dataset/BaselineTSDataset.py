from torch.utils.data import Dataset
from src.utils.data_utils.data_format_utils import get_adj_matrix_id

class BaselineTSDataset(Dataset):
    # Dataset class that can be used with the baselines PCMCI(+), VARLiNGAM and DYNOTEARS
    def __init__(self,
                 X,
                 adj_matrix,
                 lag,
                 aggregated_graph=False,
                 return_graph_indices=False):
        """
        X: np.array of shape (n_samples, timesteps, num_nodes, data_dim)
        adj_matrix: np.array of shape (n_samples, lag+1, num_nodes, num_nodes)
        """
        self.lag = lag
        self.aggregated_graph = aggregated_graph
        self.X = X
        self.adj_matrix = adj_matrix
        self.return_graph_indices = return_graph_indices
        if self.return_graph_indices:
            self.unique_matrices, self.matrix_indices = get_adj_matrix_id(self.adj_matrix)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if not self.return_graph_indices:
            return self.X[index], self.adj_matrix[index]
        return self.X[index], self.adj_matrix[index], self.matrix_indices[index]
    