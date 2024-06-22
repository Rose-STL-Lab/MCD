import torch
import numpy as np


def convert_data_to_timelagged(X, lag):
    """
    Converts data with shape (n_samples, timesteps, num_nodes, data_dim) to two tensors,
    one history tensor of shape (n_fragments, lag, num_nodes, data_dim) and
    another "input" tensor of shape (n_fragments, num_nodes, data_dim)
    """
    n_samples, timesteps, num_nodes, data_dim = X.shape

    n_fragments_per_sample = timesteps - lag
    n_fragments = n_samples * n_fragments_per_sample
    X_history = np.zeros((n_fragments, lag, num_nodes, data_dim))
    X_input = np.zeros((n_fragments, num_nodes, data_dim))
    X_indices = np.zeros((n_fragments))
    for i in range(n_fragments_per_sample):
        X_history[i*n_samples:(i+1)*n_samples] = X[:, i:i+lag, :, :]
        X_input[i*n_samples:(i+1)*n_samples] = X[:, i+lag, :, :]
        X_indices[i*n_samples:(i+1)*n_samples] = np.arange(n_samples)
    return X_history, X_input, X_indices


def get_adj_matrix_id(A):
    return np.unique(A, axis=(0), return_inverse=True)


def convert_adj_to_timelagged(A, lag, n_fragments, aggregated_graph=False, return_indices=True):
    """
    Converts adjacency matrix with shape (n_samples, (lag+1), num_nodes, num_nodes) to shape
    (n_fragments, lag+1, num_nodes, num_nodes)
    """

    A_indices = np.zeros((n_fragments))
    if len(A.shape) == 4:
        n_samples, L, num_nodes, num_nodes = A.shape
        assert L == lag+1
        Ap = np.zeros((n_fragments, lag+1, num_nodes, num_nodes))
    elif aggregated_graph:
        n_samples, num_nodes, num_nodes = A.shape
        Ap = np.zeros((n_fragments, num_nodes, num_nodes))
    else:
        assert False, "invalid adjacency matrix"
    n_fragments_per_sample = n_fragments // n_samples

    _, matrix_indices = get_adj_matrix_id(A)

    for i in range(n_fragments_per_sample):
        Ap[i*n_samples:(i+1)*n_samples] = A
        A_indices[i*n_samples:(i+1)*n_samples] = matrix_indices

    if return_indices:
        return Ap, A_indices
    return Ap


def to_time_aggregated_graph_np(graph):
    # convert graph of shape [batch, lag+1, num_nodes, num_nodes] to aggregated
    # graph of shape [batch, num_nodes, num_nodes]
    return (np.sum(graph, axis=1) > 0).astype(int)


def to_time_aggregated_scores_np(graph):
    return np.max(graph, axis=1)


def to_time_aggregated_graph_torch(graph):
    # convert graph of shape [batch, lag+1, num_nodes, num_nodes] to aggregated
    # graph of shape [batch, num_nodes, num_nodes]
    return (torch.sum(graph, dim=1) > 0).long()


def to_time_aggregated_scores_torch(graph):
    # convert edge probability matrix of shape [batch, lag+1, num_nodes, num_nodes] to aggregated
    # matrix of shape [batch, num_nodes, num_nodes]
    max_val, _ = torch.max(graph, dim=1)
    return max_val


def zero_out_diag_np(G):

    if len(G.shape) == 3:
        N = G.shape[1]
        I = np.arange(N)
        G[:, I, I] = 0

    elif len(G.shape) == 2:
        N = G.shape[0]
        I = np.arange(N)
        G[I, I] = 0

    return G


def zero_out_diag_torch(G):

    if len(G.shape) == 3:
        N = G.shape[1]
        I = torch.arange(N)
        G[:, I, I] = 0

    elif len(G.shape) == 2:
        N = G.shape[0]
        I = torch.arange(N)
        G[I, I] = 0

    return G
