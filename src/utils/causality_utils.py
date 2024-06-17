"""
Borrowed from github.com/microsoft/causica
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import torch

def convert_temporal_to_static_adjacency_matrix(
    adj_matrix: np.ndarray, conversion_type: str, fill_value: Union[float, int] = 0.0
) -> np.ndarray:
    """
    This method convert the temporal adjacency matrix to a specified type of static adjacency.
    It supports two types of conversion: "full_time" and "auto_regressive".
    The conversion type determines the connections between time steps.
    "full_time" will convert the temporal adjacency matrix to a full-time static graph, where the connections between lagged time steps are preserved.
    "auto_regressive" will convert temporal adj to a static graph that only keeps the connections to the current time step.
    E.g. a temporal adj matrix with lag 2 is [A,B,C], where A,B and C are also adj matrices. "full_time" will convert this to
    [[A,B,C],[0,A,B],[0,0,A]]. "auto_regressive" will convert this to [[0,0,C],[0,0,B],[0,0,A]].
    "fill_value" is used to specify the value to fill in the converted static adjacency matrix. Default is 0, but sometimes we may want
    other values. E.g. if we have a temporal soft prior with prior mask, then we may want to fill the converted prior mask with value 1 instead of 0,
    since the converted prior mask should never disable the blocks specifying the "arrow-against-time" in converted soft prior.
    Args:
        adj_matrix: The temporal adjacency matrix with shape [lag+1, from, to] or [N, lag+1, from, to].
        conversion_type: The conversion type. It supports "full_time" and "auto_regressive".
        fill_value: The value used to fill the static adj matrix. The default is 0.

    Returns: static adjacency matrix with shape [(lag+1)*from, (lag+1)*to] or [N, (lag+1)*from, (lag+1)*to].

    """
    assert conversion_type in [
        "full_time",
        "auto_regressive",
    ], f"The conversion_type {conversion_type} is not supported."
    if len(adj_matrix.shape) == 3:
        adj_matrix = adj_matrix[None, ...]  # [1, lag+1, num_node, num_node]
    batch_dim, n_lag, n_nodes, _ = adj_matrix.shape  # n_lag is lag+1
    if conversion_type == "full_time":
        block_fill_value = np.full((n_nodes, n_nodes), fill_value)
    else:
        block_fill_value = np.full(
            (batch_dim, n_lag * n_nodes, (n_lag - 1) * n_nodes), fill_value)

    if conversion_type == "full_time":
        static_adj = np.sum(
            np.stack([np.kron(np.diag(np.ones(n_lag - i), k=i),
                     adj_matrix[:, i, :, :]) for i in range(n_lag)], axis=1),
            axis=1,
        )  # [N, n_lag*from, n_lag*to]
        static_adj += np.kron(
            np.tril(np.ones((batch_dim, n_lag, n_lag)), k=-1), block_fill_value
        )  # [N, n_lag*from, n_lag*to]

    if conversion_type == "auto_regressive":
        # Flip the temporal adj and concatenate to form one block column of the static. The flipping is needed due to the
        # format of converted adjacency matrix. E.g. temporal adj [A,B,C], where A is the instant adj matrix. Then, the converted adj
        # is [[[0,0,C],[0,0,B],[0,0,A]]]. The last column is the concatenation of flipped temporal adj.
        block_column = np.flip(adj_matrix, axis=1).reshape(
            -1, n_lag * n_nodes, n_nodes
        )  # [N, (lag+1)*num_node, num_node]
        # Static graph
        # [N, (lag+1)*num_node, (lag+1)*num_node]
        static_adj = np.concatenate((block_fill_value, block_column), axis=2)

    return np.squeeze(static_adj)


def dag_pen_np(X):
    assert X.shape[0] == X.shape[1]
    X = torch.from_numpy(X)
    return (torch.trace(torch.matrix_exp(X)) - X.shape[0]).item()


def approximate_maximal_acyclic_subgraph(adj_matrix: np.ndarray, n_samples: int = 10) -> np.ndarray:
    """
    Compute an (approximate) maximal acyclic subgraph of a directed non-dag but removing at most 1/2 of the edges
    See Vazirani, Vijay V. Approximation algorithms. Vol. 1. Berlin: springer, 2001, Page 7;
    Also Hassin, Refael, and Shlomi Rubinstein. "Approximations for the maximum acyclic subgraph problem."
    Information processing letters 51.3 (1994): 133-140.
    Args:
        adj_matrix: adjacency matrix of a directed graph (may contain cycles)
        n_samples: number of the random permutations generated. Default is 10.
    Returns:
        an adjacency matrix of the acyclic subgraph
    """
    # assign each node with a order
    adj_dag = np.zeros_like(adj_matrix)
    for _ in range(n_samples):
        random_order = np.expand_dims(
            np.random.permutation(adj_matrix.shape[0]), 0)
        # subgraph with only forward edges defined by the assigned order
        adj_forward = (
            (random_order.T > random_order).astype(int)) * adj_matrix
        # subgraph with only backward edges defined by the assigned order
        adj_backward = (
            (random_order.T < random_order).astype(int)) * adj_matrix
        # return the subgraph with the least deleted edges
        adj_dag_n = adj_forward if adj_backward.sum() < adj_forward.sum() else adj_backward
        if adj_dag_n.sum() > adj_dag.sum():
            adj_dag = adj_dag_n
    return adj_dag


def int2binlist(i: int, n_bits: int):
    """
    Convert integer to list of ints with values in {0, 1}
    """
    assert i < 2**n_bits
    str_list = list(np.binary_repr(i, n_bits))
    return [int(i) for i in str_list]


def cpdag2dags(cp_mat: np.ndarray, samples: Optional[int] = None) -> np.ndarray:
    """
    Compute all possible DAGs contained within a Markov equivalence class, given by a CPDAG
    Args:
        cp_mat: adjacency matrix containing both forward and backward edges for edges for which directionality is undetermined
    Returns:
        3 dimensional tensor, where the first indexes all the possible DAGs
    """
    assert len(cp_mat.shape) == 2 and cp_mat.shape[0] == cp_mat.shape[1]

    # matrix composed of just undetermined edges
    cycle_mat = (cp_mat == cp_mat.T) * cp_mat
    # return original matrix if there are no length-1 cycles
    if cycle_mat.sum() == 0:
        if dag_pen_np(cp_mat) != 0.0:
            cp_mat = approximate_maximal_acyclic_subgraph(cp_mat)
        return cp_mat[None, :, :]

    # matrix of determined edges
    cp_determined_subgraph = cp_mat - cycle_mat

    # prune cycles if the matrix of determined edges is not a dag
    if dag_pen_np(cp_determined_subgraph.copy()) != 0.0:
        cp_determined_subgraph = approximate_maximal_acyclic_subgraph(
            cp_determined_subgraph, 1000)

    # number of parent nodes for each node under the well determined matrix
    n_in_nodes = cp_determined_subgraph.sum(axis=0)

    # lower triangular version of cycles edges: only keep cycles in one direction.
    cycles_tril = np.tril(cycle_mat, k=-1)

    # indices of potential new edges
    undetermined_idx_mat = np.array(np.nonzero(
        cycles_tril)).T  # (N_undedetermined, 2)

    # number of undetermined edges
    N_undetermined = int(cycles_tril.sum())

    # choose random order for mask iteration
    max_dags = 2**N_undetermined

    if max_dags > 10000:
        print("The number of possible dags are too large (>10000), limit to 10000")
        max_dags = 10000

    if samples is None:
        samples = max_dags
    mask_indices = list(np.random.permutation(np.arange(max_dags)))

    # iterate over list of all potential combinations of new edges. 0 represents keeping edge from upper triangular and 1 from lower triangular
    dag_list: list = []
    while mask_indices and len(dag_list) < samples:

        mask_index = mask_indices.pop()
        mask = np.array(int2binlist(mask_index, N_undetermined))

        # extract list of indices which our new edges are pointing into
        incoming_edges = np.take_along_axis(
            undetermined_idx_mat, mask[:, None], axis=1).squeeze()

        # check if multiple edges are pointing at same node
        _, unique_counts = np.unique(
            incoming_edges, return_index=False, return_inverse=False, return_counts=True)

        # check if new colider has been created by checkig if multiple edges point at same node or if new edge points at existing child node
        new_colider = np.any(unique_counts > 1) or np.any(
            n_in_nodes[incoming_edges] > 0)

        if not new_colider:
            # get indices of new edges by sampling from lower triangular mat and upper triangular according to indices
            edge_selection = undetermined_idx_mat.copy()
            edge_selection[mask == 0, :] = np.fliplr(
                edge_selection[mask == 0, :])

            # add new edges to matrix and add to dag list
            new_dag = cp_determined_subgraph.copy()
            new_dag[(edge_selection[:, 0], edge_selection[:, 1])] = 1

            # Check for high order cycles
            if dag_pen_np(new_dag.copy()) == 0.0:
                dag_list.append(new_dag)
    # When all combinations of new edges create cycles, we will only keep determined ones
    if len(dag_list) == 0:
        dag_list.append(cp_determined_subgraph)

    return np.stack(dag_list, axis=0)
