import os
import argparse
import yaml

from data_generation_utils import generate_cts_temporal_data, set_random_seed, generate_temporal_graph
import cdt
import numpy as np
import networkx as nx


def calc_dist(adj_matrix):
    unique_matrices = np.unique(adj_matrix, axis=0)
    distances = []
    for i in range(unique_matrices.shape[0]):
        for j in range(i):
            distances.append(cdt.metrics.SHD(
                unique_matrices[i], unique_matrices[j]))
    mean_dist = np.mean(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    std_dist = np.std(distances)

    return mean_dist, std_dist, min_dist, max_dist


def perturb_graph(G, p):
    retry_counter = 0
    while True:
        perturbation = np.random.binomial(1, p, size=G.shape)
        perturbed = np.logical_xor(G, perturbation).astype(int)
        # nxG = nx.from_numpy_matrix(perturbed[0], create_using=nx.DiGraph)
        nxG = nx.DiGraph(perturbed[0])
        if nx.is_directed_acyclic_graph(nxG):
            break
        retry_counter += 1

        if retry_counter >= 200000:
            assert False, "Cannot generate DAG, try a lower value of p"

    return perturbed.astype(int)


def main(config_file):

    # read the yaml file
    with open(config_file, 'r', encoding="utf-8") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)

    series_length = int(data_config["num_timesteps"])
    burnin_length = int(data_config["burnin_length"])
    num_samples = int(data_config["num_samples"])
    disable_inst = bool(data_config["disable_inst"])
    graph_type = [data_config["inst_graph_type"],
                  data_config["lag_graph_type"]]
    p_array = data_config['p_array']

    connection_factor = 1

    # in this case, all inst and lagged adj are dags and have total edges per adj = 2*num_nodes
    # graph_type = ["SF", "SF"]
    # graph_config = [{"m":2, "directed":True}, {"m":2, "directed":True}]

    lag = int(data_config["lag"])
    is_history_dep = bool(data_config["history_dep_noise"])

    noise_level = float(data_config["noise_level"])
    function_type = data_config["function_type"]
    noise_function_type = data_config["noise_function_type"]
    base_noise_type = data_config["base_noise_type"]

    save_dir = data_config["save_dir"]
    N = data_config["num_nodes"]
    num_graphs = data_config["num_graphs"]

    for seed in data_config['random_seed']:
        seed = int(seed)
        set_random_seed(seed)
        graph_config = [
            {"m": N * 2 * connection_factor if not disable_inst else 0, "directed": True},
            {"m": N * connection_factor, "directed": True},
        ]
        G = generate_temporal_graph(
            N, graph_type, graph_config, lag=2).astype(int)
        N = int(N)

        for N_G in num_graphs:
            N_G = int(N_G)
            print(f"Generating dataset for N={N}, num_graphs={N_G}")

            for p in p_array:
                graphs = []
                for i in range(N_G):
                    print(f"Generating graph {i}/{N_G}")
                    Gtilde = perturb_graph(G, p)
                    graphs.append(Gtilde)

                mean_dist, std_dist, min_dist, max_dist = calc_dist(
                    np.array(graphs))
                print(
                    f"Perturbation,{N},{N_G},{mean_dist},{std_dist},{min_dist},{max_dist},{p}")

                folder_name = f"perturb_N{N}_K{N_G}_p{p}_seed{seed}"
                path = os.path.join(save_dir, folder_name)

                generate_cts_temporal_data(
                    path=path,
                    num_graphs=N_G,
                    series_length=series_length,
                    burnin_length=burnin_length,
                    num_samples=num_samples,
                    num_nodes=N,
                    graph_type=graph_type,
                    graph_config=graph_config,
                    lag=lag,
                    is_history_dep=is_history_dep,
                    noise_level=noise_level,
                    function_type=function_type,
                    noise_function_type=noise_function_type,
                    base_noise_type=base_noise_type,
                    temporal_graphs=graphs
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Temporal Synthetic Data Generator")
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    main(config_file=args.config_file)
