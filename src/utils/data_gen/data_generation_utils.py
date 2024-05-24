"""
Borrowed from github.com/microsoft/causica
"""


import os
import random
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple

import igraph as ig
import numpy as np
import torch
from pyro.distributions.transforms.spline import ConditionalSpline
from pyro.nn.dense_nn import DenseNN
from tqdm import tqdm
from torch import nn
from splines import unconstrained_RQS


class PiecewiseRationalQuadraticTransform(nn.Module):
    """
    Layer that implements a spline-cdf (https://arxiv.org/abs/1906.04032) transformation.
     All dimensions of x are treated as independent, no coupling is used. This is needed
    to ensure invertibility in our additive noise SEM.

    Args:
        dim: dimensionality of input,
        num_bins: how many bins to use in spline,
        tail_bound: distance of edgemost bins relative to 0,
        init_scale: standard deviation of Gaussian from which spline parameters are initialised
    """

    def __init__(
        self,
        dim,
        num_bins=8,
        tail_bound=3.0,
        init_scale=1e-2,
    ):
        super().__init__()

        self.dim = dim
        self.num_bins = num_bins
        self.min_bin_width = 1e-3
        self.min_bin_height = 1e-3
        self.min_derivative = 1e-3
        self.tail_bound = tail_bound
        self.init_scale = init_scale

        self.params = nn.Parameter(
            self.init_scale * torch.randn(self.dim, self.num_bins * 3 - 1), requires_grad=True)

    def _piecewise_cdf(self, inputs, inverse=False):
        params_batch = self.params.unsqueeze(
            dim=(0)).expand(inputs.shape[0], -1, -1)

        unnormalized_widths = params_batch[..., : self.num_bins]
        unnormalized_heights = params_batch[...,
                                            self.num_bins: 2 * self.num_bins]
        unnormalized_derivatives = params_batch[..., 2 * self.num_bins:]

        return unconstrained_RQS(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            tail_bound=self.tail_bound,
        )

    def forward(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        return self._piecewise_cdf(inputs, inverse=False)

    def inverse(self, inputs):
        """
        Args:
            input: (batch_size, input_dim)
        Returns:
            transformed_input, jacobian_log_determinant: (batch_size, input_dim), (batch_size, input_dim)
        """
        return self._piecewise_cdf(inputs, inverse=True)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_permutation(M: np.ndarray) -> np.ndarray:
    """
    This will randomly permute the matrix M.
    Args:
        M: the input matrix with shape [num_node, num_node].

    Returns:
        The permuted matrix
    """
    P = np.random.permutation(np.eye(M.shape[0]))
    return P.T @ M @ P


def random_acyclic_orientation(B_und: np.ndarray) -> np.ndarray:
    """
    This will randomly permute the matrix B_und followed by taking the lower triangular part.
    Args:
        B_und: The input matrix with shape [num_node, num_node].

    Returns:
        The lower triangular of the permuted matrix.
    """
    return np.tril(random_permutation(B_und), k=-1)


def generate_single_graph(num_nodes: int, graph_type: str, graph_config: dict, is_DAG: bool = True) -> np.ndarray:
    """
    This will generate a single adjacency matrix following different graph generation methods (specified by graph_type, can be "ER", "SF", "SBM").
    graph_config specifes the additional configurations for graph_type. For example, for "ER", the config dict keys can be {"p", "m", "directed", "loop"},
    refer to igraph for details. is_DAG is to ensure the generated graph is a DAG by lower-trianguler the adj, followed by a permutation.
    Note that SBM will no longer be a proper SBM if is_DAG=True
    Args:
        num_nodes: The number of nodes
        graph_type: The graph generation type. "ER", "SF" or "SBM".
        graph_config: The dict containing additional argument for graph generation.
        is_DAG: bool indicates whether the generated graph is a DAG or not.

    Returns:
        An binary ndarray with shape [num_node, num_node]
    """
    if graph_type == "ER":
        adj_graph = np.array(ig.Graph.Erdos_Renyi(
            n=num_nodes, **graph_config).get_adjacency().data)
        if is_DAG:
            adj_graph = random_acyclic_orientation(adj_graph)
    elif graph_type == "SF":
        if is_DAG:
            graph_config["directed"] = True
        adj_graph = np.array(ig.Graph.Barabasi(
            n=num_nodes, **graph_config).get_adjacency().data)
    elif graph_type == "SBM":
        adj_graph = np.array(ig.Graph.SBM(
            n=num_nodes, **graph_config).get_adjacency().data)
        if is_DAG:
            adj_graph = random_acyclic_orientation(adj_graph)
    else:
        raise ValueError("Unknown graph type")

    if is_DAG or graph_type == "SF":
        # SF needs to be permuted otherwise it generates either lowtri or symmetric matrix.
        adj_graph = random_permutation(adj_graph)

    return adj_graph


def generate_temporal_graph(
    num_nodes: int, graph_type: List[str], graph_config: List[dict], lag: int, random_state: Optional[int] = None
) -> np.ndarray:
    """
    This will generate a temporal graph with shape [lag+1, num_nodes, num_nodes] based on the graph_type. The graph_type[0] specifies the
    generation type for instantaneous effect and graph_type[1] specifies the lagged effect. For re-produciable results, set random_state.
    Args:
        num_nodes: The number of nodes.
        graph_type: A list containing the graph generation type. graph_type[0] for instantaneous effect and graph_type[1] for lagged effect.
        graph_config: A list of dict containing the configs for graph generation. The order should respect the graph_type.
        lag: The lag of the graph.
        random_state: The random seed used to generate the graph. None means not setting the seed.

    Returns:
        temporal_graph with shape [lag+1, num_nodes, num_nodes]
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    temporal_graph = np.full([lag + 1, num_nodes, num_nodes], np.nan)
    # Generate instantaneous effect graph
    temporal_graph[0] = generate_single_graph(
        num_nodes, graph_type[0], graph_config[0], is_DAG=True)
    # Generate lagged effect graph
    for i in range(1, lag + 1):
        temporal_graph[i] = generate_single_graph(
            num_nodes, graph_type[1], graph_config[1], is_DAG=False)

    return temporal_graph


def extract_parents(data: np.ndarray, temporal_graph: np.ndarray, node: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function will extract the parent values from data with given graph temporal_graph. It will return the lagged parents
    and instantaneous parents.
    Args:
        data: ndarray with shape [series_length, num_nodes] or [batch, series_length, num_nodes]
        temporal_graph: A binary ndarray with shape [lag+1, num_nodes, num_nodes]

    Returns:
        instant_parent: instantaneous parents with shape [parent_dim] or [batch, parents_dim]
        lagged_parent: lagged parents with shape [lagged_dim] or [batch, lagged_dim]
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # shape [1, series_length, num_nodes]

    assert data.ndim == 3, "data should be of shape [series_length, num_nodes] or [batch, series_length, num_nodes]"
    lag = temporal_graph.shape[0] - 1
    # extract instantaneous parents
    inst_parent_node = temporal_graph[0, :, node].astype(
        bool)  # shape [num_parents]
    # shape [batch, parent_dim]
    inst_parent_value = data[:, -1, inst_parent_node]

    # Extract lagged parents
    lagged_parent_value_list = []
    for cur_lag in range(1, lag + 1):
        cur_lag_parent_node = temporal_graph[cur_lag, :, node].astype(
            bool)  # shape [num_parents]
        # shape [batch, parent_dim]
        cur_lag_parent_value = data[:, -cur_lag - 1, cur_lag_parent_node]
        lagged_parent_value_list.append(cur_lag_parent_value)

    # shape [batch, lagged_dim_aggregated]
    lagged_parent_value = np.concatenate(lagged_parent_value_list, axis=1)

    # if data.shape[0] == 1:
    #     inst_parent_value, lagged_parent_value = inst_parent_value.squeeze(
    #         0), lagged_parent_value.squeeze(0)

    return inst_parent_value, lagged_parent_value


def simulate_history_dep_noise(lagged_parent_value: np.ndarray, noise: np.ndarray, noise_func: Callable) -> np.ndarray:
    """
    This will simulate the history-dependent noise given the lagged parent value.
    Args:
        lagged_parent_value: ndarray with shape [batch, lag_parent_dim] or [lag_parent_dim]
        noise: ndarray with shape [batch,1] or [1]
        noise_func: this specifies the function transformation for noise

    Returns:
        history-dependent noise with shape [batch, 1] or [1]
    """

    assert (
        lagged_parent_value.shape[0] == noise.shape[0]
    ), "lagged_parent_value and noise should have the same batch size"
    if lagged_parent_value.ndim == 1:
        # shape [1, lag_parent_dim]
        lagged_parent_value = lagged_parent_value[np.newaxis, ...]
        noise = noise[np.newaxis, ...]  # [1, 1]

    # concat the lagged parent value and noise
    # shape [batch, lag_parent_dim+1]
    input_to_gp = np.concatenate([lagged_parent_value, noise], axis=1)
    history_dependent_noise = noise_func(input_to_gp)  # shape [batch, 1]

    if lagged_parent_value.shape[0] == 1:
        history_dependent_noise = history_dependent_noise.squeeze(0)

    return history_dependent_noise


def simulate_function(lag_inst_parent_value: np.ndarray, func: Callable) -> np.ndarray:
    """
    This will simulate the function given the lagged and instantaneous parent values. The random_state_value controls
    which function is sampled from gp_func, and it controls which function is used for a particular node.
    Args:
        lag_inst_parent_value: ndarray with shape [batch, lag+inst_parent_dim] or [lag+inst_parent_dim]
        random_state_value: int controlling the function sampled from gp
        func: This specifies the functional relationships

    Returns:
        ndarray with shape [batch, 1] or [1] representing the function value for the current node.
    """

    if lag_inst_parent_value.ndim == 1:
        # shape [1, lag+inst_parent_dim]
        lag_inst_parent_value = lag_inst_parent_value[np.newaxis, ...]

    function_value = func(lag_inst_parent_value)  # shape [batch, 1]

    if lag_inst_parent_value.shape[0] == 1:
        function_value = function_value.squeeze(0)

    return function_value


def simulate_single_step(
    history_data: np.ndarray,
    temporal_graph: np.ndarray,
    func_list_noise: List[Callable],
    func_list: List[Callable],
    topological_order: List[int],
    is_history_dep: bool = False,
    noise_level: float = 1,
    base_noise_type: str = "gaussian",
    intervention_dict: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    This will generate the data for a particular timestamp given the history data and temporal graph.
    Args:
        history_data: History data with shape [batch, series_length, num_node] or [series_length, num_node] containing the history observations.
        temporal_graph: The binary ndarray graph with shape [lag+1, num_nodes, num_nodes].
        func_list_noise: List of function transforms for the noise variable. Shape [num_nodes]. Each function takes [batch, dim] as input and
        output [batch,1]
        func_list: List of function for each variable, shape [num_nodes]. Each func takes [batch, dim] as input and output [batch,1]
        topological_order: the topological order from source to leaf nodes, specified by temporal graph.
        is_history_dep: bool indicate if the noise is history dependent
        base_noise_type: str, support "gaussian" and "uniform"
        intervention_dict: dict holding interventions for the current time step of form {intervention_idx: intervention_value}
    Returns:
        ndarray with shape [batch,num_node] or [num_node]
    """

    assert (
        len(func_list_noise) == len(func_list) == temporal_graph.shape[1]
    ), "function list and topological_order should have the same length as the number of nodes"
    if history_data.ndim == 2:
        # shape [1, series_length, num_nodes]
        history_data = history_data[np.newaxis, ...]

    if intervention_dict is None:
        intervention_dict = {}

    batch_size = history_data.shape[0]
    # iterate through the nodes in topological order
    for node in topological_order:
        # if this node is intervened on - ignore the functions and parents
        if node in intervention_dict:
            history_data[:, -1, node] = intervention_dict[node]
        else:
            # extract the instantaneous and lagged parent values
            inst_parent_value, lagged_parent_value = extract_parents(
                history_data, temporal_graph, node
            )  # [batch, inst_parent_dim], [batch, lagged_dim_aggregated]

            # simulate the noise
            if base_noise_type == "gaussian":
                Z = np.random.randn(history_data.shape[0], 1)  # [batch, 1]
            elif base_noise_type == "uniform":
                Z = np.random.rand(history_data.shape[0], 1)
            else:
                raise NotImplementedError
            if is_history_dep:
                # Extract the noise transform
                noise_func = func_list_noise[node]

                Z = simulate_history_dep_noise(
                    lagged_parent_value, Z, noise_func)  # [batch, 1]

            # simulate the function
            lag_inst_parent_value = np.concatenate(
                [inst_parent_value, lagged_parent_value], axis=1
            )  # [batch, lag+inst_parent_dim]
            if lag_inst_parent_value.size == 0:
                func_value = np.array(0.0)
            else:
                # Extract the function relation
                func = func_list[node]
                func_value = simulate_function(lag_inst_parent_value, func)
            # [batch]
            history_data[:, -1,
                         node] = (func_value + noise_level * Z).squeeze(-1)

    X = history_data[:, -1, :]  # [batch,num_nodes]
    if batch_size == 1:
        X = X.squeeze(0)
    return X


def format_data(data: np.ndarray):
    """
    This will transform the data format to be compatiable with the temporal data loader. It will add additional column to
    indicate the series number.
    Args:
        data: ndarray with shape [batch, series_length, num_node]

    Returns:
        transformed_data with shape [batch*series_length, num_node+1]
    """
    batch_size = data.shape[0]
    transfomred_data_list = []
    for cur_batch_idx in range(batch_size):
        cur_batch = data[cur_batch_idx, ...]  # [series_length, num_node]

        cur_series_idx = cur_batch_idx * \
            np.ones((cur_batch.shape[0], 1))  # [series_length, 1]
        cur_transformed_batch = np.concatenate(
            [cur_series_idx, cur_batch], axis=1)  # [series_length, num_node+1]

        transfomred_data_list.append(cur_transformed_batch)
        # transfomred_data_list.append(cur_transformed_batch)
    # [batch*series_length, num_node+1]
    transformed_data = np.concatenate(transfomred_data_list, axis=0)
    return transformed_data


def select_random_history(timeseries: np.ndarray, lag: int, num_samples: int = 1) -> np.ndarray:
    """Selects random history conditioning from the input timeseries. This selects a single window per input timeseries.

    Args:
        timeseries (np.ndarray): The timeseries to select subsets from. Should have shape [batch, time, variables]
        lag (int): The window length to select.
        num_samples (int): How many histories to select. This should be <= batch.

    Returns:
        np.ndarray: Selected histories of shape [num_samples, lag, variables]
    """
    # adapted this from here: https://stackoverflow.com/questions/47982894/selecting-random-windows-from-multidimensional-numpy-array-rows
    b, t, n = timeseries.shape
    assert num_samples <= b
    idx = np.random.randint(0, t - lag + 1, b)

    s0, s1, s2 = timeseries.strides
    windows = np.lib.stride_tricks.as_strided(timeseries, shape=(
        b, t - lag + 1, lag, n), strides=(s0, s1, s1, s2))
    if num_samples < b:
        idx2 = np.random.randint(0, b, num_samples)
        return windows[np.arange(len(idx))[idx2], idx[idx2], :, :]
    return windows[np.arange(len(idx)), idx, :, :]


def generate_cts_temporal_data(
    path: str,
    series_length: int,
    burnin_length: int,
    num_samples: int,
    num_nodes: int,
    graph_type: List[str],
    graph_config: List[dict],
    lag: int,
    num_graphs: int = 1,
    is_history_dep: bool = False,
    noise_level: float = 1,
    function_type: str = "spline",
    noise_function_type: str = "spline",
    save_data: bool = True,
    base_noise_type: str = "gaussian",
    temporal_graphs = None
) -> np.ndarray:
    """
    This will generate continuous time-series data (with history-depdendent noise). It will start to collect the data after the burnin_length for stationarity.
    Args:
        path: The output dir path.
        series_length: The time series length to be generated.
        burnin_length: The burnin length before collecting the data.
        num_train_samples: The batch size of the time series data.
        num_test_samples: Number of test samples
        num_nodes: the number of variables within each timestamp.
        graph_type: The list of str specifying the instant graph and lagged graph types. graph_type[0] is for instant and
        graph_type[1] is for lagged. The choices are "ER", "SF" and "SBM".
        graph_config: A list of dict containing the graph generation configs. It should respect the order in graph_type.
        is_history_dep: bool to indicate whether the history-dependent noise is considered. Otherwise, it is Gaussian noise.
        noise_level: the std of the Gaussian noise.
        function_type: the type of function to be used for SEM. Support "spline"
        noise_function_type: the type of function to be used for history dependent noise. Support "spline_product", "conditional_spline".
        save_data: whether to save the data.
        base_noise_type: str, the base noise distribution, supports "gaussian" and "uniform"
    Returns:
        None, but the stored ndarray has shape [batch*series_length, num_nodes+1], where the +1 is index of time series.
    """

    num_all_samples = num_samples

    if temporal_graphs is not None:
        assert len(temporal_graphs) == num_graphs
    else:
        temporal_graphs = []

    func_lists = []
    noise_func_lists = []
    single_steps = []
    graph_list = []
    X_by_graph = [[] for i in range(num_graphs)]

    # Generate graphs
    for i in range(num_graphs):
        if temporal_graphs is None or len(temporal_graphs) < num_graphs:
            temporal_graph = generate_temporal_graph(
                num_nodes=num_nodes, graph_type=graph_type, graph_config=graph_config, lag=lag, random_state=None
            )
            temporal_graphs.append(temporal_graph)
        else:
            temporal_graph = temporal_graphs[i]

        # Build the function and noise_function list
        func_list, noise_func_list = build_function_list(
            temporal_graph, function_type=function_type, noise_function_type=noise_function_type
        )
        func_lists.append(func_list)
        noise_func_lists.append(noise_func_list)
        # Find topological order of instant graph
        ig_graph = ig.Graph.Adjacency(temporal_graph[0].tolist())
        topological_order = ig_graph.topological_sorting()
        single_step = partial(
            simulate_single_step,
            temporal_graph=temporal_graph,
            func_list=func_list,
            func_list_noise=noise_func_list,
            topological_order=topological_order,
            is_history_dep=is_history_dep,
            noise_level=noise_level,
            base_noise_type=base_noise_type,
        )
        single_steps.append(single_step)

    # Start data gen
    X_all = np.full((num_all_samples, burnin_length +
                    series_length + lag, num_nodes), np.nan)
    X_all[..., 0:lag, :] = (
        np.random.randn(num_all_samples, lag,
                        num_nodes) if num_all_samples > 1 else np.random.randn(lag, num_nodes)
    )

    for i in tqdm(range(num_all_samples)):
        # randomly choose a graph
        graph_choice = random.randint(0, num_graphs-1)
        graph_list.append(temporal_graphs[graph_choice])
        # print("Chose graph:", temporal_graphs[graph_choice])
        for time in range(lag, burnin_length + series_length + lag):
            single_step = single_steps[graph_choice]
            # X_all[i, time, :] = np.squeeze(single_step(
            # history_data=np.expand_dims(X_all[i, time - lag: time + 1, :], axis=0)), axis=0)
            X_all[i, time, :] = single_step(
                history_data=X_all[i, time - lag: time + 1, :])
        X_by_graph[graph_choice].append(X_all[i, lag+burnin_length:, :])

    # extract the stationary part of the data
    # shape [batch, series_length, num_nodes]
    X_stationary = X_all[..., lag + burnin_length:, :]

    if X_stationary.ndim == 2:
        X_stationary = X_stationary[np.newaxis]

    graph_list = np.array(graph_list)
    print(graph_list.shape)
    # Save the data
    if save_data:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(
                "[WARNING]: The save directory already exists. This might lead to unexpected behaviour.")

        np.savetxt(os.path.join(path, "train.csv"),
                   format_data(X_stationary), delimiter=",")
        np.save(os.path.join(path, "X.npy"), X_stationary)
        # Save the adjacency_matrix
        np.save(os.path.join(path, "adj_matrix.npy"), graph_list)

    # save by-graph data
    path = os.path.join(path, 'grouped_by_graph')
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(num_graphs):
        X = np.array(X_by_graph[i])
        adj_matrix = np.array(temporal_graphs[i])
        np.save(os.path.join(path, f"X_{i}.npy"), X)
        # Save the adjacency_matrix
        np.save(os.path.join(path, f"adj_matrix_{i}.npy"), adj_matrix)
    return X_stationary


def build_function_list(
    temporal_graph: np.ndarray, function_type: str, noise_function_type: str
) -> Tuple[List[Callable], List[Callable]]:
    """
    This will build two lists containing the SEM functions and history-dependent noise function, respectively.
    Args:
        temporal_graph: The input temporal graph.
        function_type: The type of SEM function used.
        noise_function_type: The tpe of history-dependent noise transformation used.

    Returns:
        function_list: list of SEM functions
        noise_function_list: list of history-dependent noise transformation
    """
    num_nodes = temporal_graph.shape[1]
    # get func_list
    function_list = []
    for cur_node in range(num_nodes):
        # get input dim
        input_dim = sum(temporal_graph[lag, :, cur_node] for lag in range(
            temporal_graph.shape[0])).sum().astype(int)  # type: ignore

        if input_dim == 0:
            function_list.append(zero_func)
        else:
            function_list.append(sample_function(
                input_dim, function_type=function_type))
    # get noise_function_list
    noise_function_list = []
    for cur_node in range(num_nodes):
        # get input dim
        input_dim = (
            sum(temporal_graph[lag, :, cur_node] for lag in range(
                1, temporal_graph.shape[0])).sum() + 1  # type: ignore
        ).astype(int)
        noise_function_list.append(sample_function(
            input_dim, function_type=noise_function_type))

    return function_list, noise_function_list


def sample_function(input_dim: int, function_type: str) -> Callable:
    """
    This will sample a function given function type.
    Args:
        input_dim: The input dimension of the function.
        function_type: The type of function to be sampled.
    Returns:
        A function sampled.
    """
    if function_type == "spline":
        return sample_spline(input_dim)
    if function_type == "spline_product":
        return sample_spline_product(input_dim)
    elif function_type == "conditional_spline":
        return sample_conditional_spline(input_dim)
    elif function_type == "mlp":
        return sample_mlp(input_dim)
    elif function_type == "inverse_noise_spline":
        return sample_inverse_noise_spline(input_dim)
    elif function_type == "mlp_noise":
        return sample_mlp_noise(input_dim)
    elif function_type == 'linear':
        return sample_linear(input_dim)
    else:
        raise ValueError(f"Unsupported function type: {function_type}")


def sample_inverse_noise_spline(input_dim):
    flow = PiecewiseRationalQuadraticTransform(1, 16, 5, 1)
    W = np.ones((input_dim - 1, 1)) * (1 / (input_dim - 1))

    def func(X):
        z = X[..., :-1] @ W
        with torch.no_grad():
            y, _ = flow(torch.from_numpy(z))

            return -y.numpy() * X[..., -1:]

    return func


def sample_conditional_spline(input_dim):
    # input_dim is lagged_parent + 1
    noise_dim = 1
    count_bins = 8
    param_dim = [noise_dim * count_bins, noise_dim *
                 count_bins, noise_dim * (count_bins - 1)]
    hypernet = DenseNN(input_dim - 1, [20, 20], param_dim)
    transform = ConditionalSpline(
        hypernet, noise_dim, count_bins=count_bins, order="quadratic")

    def func(X):
        """
        X: lagged parents concat with noise. X[...,0:-1] lagged parents, X[...,-1] noise.
        """
        with torch.no_grad():
            X = torch.from_numpy(X).float()
            transform_cond = transform.condition(X[..., :-1])
            noise_trans = transform_cond(X[..., -1:])  # [batch, 1]
        return noise_trans.numpy()

    return func


def sample_spline(input_dim):
    flow = PiecewiseRationalQuadraticTransform(1, 16, 5, 1)
    W = np.ones((input_dim, 1)) * (1 / input_dim)

    def func(X):
        z = X @ W
        with torch.no_grad():
            y, _ = flow(torch.from_numpy(z))
            return y.numpy()

    return func


def sample_mlp(input_dim):
    mlp = DenseNN(input_dim, [64, 64], [1])

    def func(X):
        X = torch.from_numpy(X).float()
        with torch.no_grad():
            return mlp(X).numpy()

    return func


def sample_mlp_noise(input_dim):
    mlp = DenseNN(input_dim - 1, [64, 64], [1])

    def func(X):
        X_pa = torch.from_numpy(X[..., :-1]).float()
        with torch.no_grad():
            return mlp(X_pa).numpy() * X[..., -1:]

    return func


def sample_spline_product(input_dim):
    flow = sample_spline(input_dim - 1)

    def func(X):
        z_p = flow(X[..., :-1])
        out = z_p * X[..., -1:]
        return out

    return func

def sample_linear(input_dim):
    # sample weights
    W = np.random.binomial(n=1, p=0.5, size=(input_dim))*np.random.uniform(0.1, 0.5, size=(input_dim))
    
    def func(X):
        return X@W

    return func

def zero_func() -> np.ndarray:
    return np.zeros(1)


def generate_name(
    num_nodes: int,
    n_samples: int,
    graph_type: List[str],
    num_graphs: int,
    lag: int,
    is_history_dep: bool,
    noise_level: float,
    function_type: str,
    noise_function_type: str,
    seed: int,
    disable_inst: bool,
    connection_factor: int,
    base_noise_type: str
) -> str:
    if not is_history_dep:
        flag = "NoHistDep"
        noise_function_type = base_noise_type
    else:
        flag = "HistDep"


    if disable_inst:
        file_name = (
            f"{graph_type[0]}_{graph_type[1]}_num_graphs_{num_graphs}_lag_{lag}_dim_{num_nodes}_{flag}_{noise_level}_{function_type}_"
            + f"{noise_function_type}_NoInst_con_{connection_factor}_seed_{seed}_n_samples_{n_samples}"
        )
    else:
        file_name = (
            f"{graph_type[0]}_{graph_type[1]}_num_graphs_{num_graphs}_lag_{lag}_dim_{num_nodes}_{flag}_{noise_level}_{function_type}_"
            + f"{noise_function_type}_con_{connection_factor}_seed_{seed}_n_samples_{n_samples}"
        )

    return file_name
