import argparse
import yaml
import os
import json

from data_generation_utils import generate_cts_temporal_data, generate_name, set_random_seed


def main(config_file):

    # read the yaml file
    with open(config_file) as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    
    series_length = int(data_config["num_timesteps"])
    burnin_length = int(data_config["burnin_length"])
    num_samples = int(data_config["num_samples"])
    disable_inst = bool(data_config["disable_inst"])
    graph_type = [data_config["inst_graph_type"], data_config["lag_graph_type"]]

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
    num_nodes = data_config["num_nodes"]
    num_graphs = data_config["num_graphs"]

    for seed in data_config['random_seed']:
        seed = int(seed)
        set_random_seed(seed)

        for N in num_nodes:
            for N_G in num_graphs:
                print(f"Generating dataset for N={N}, num_graphs={N_G}")
                N = int(N)
                N_G = int(N_G)
                graph_config = [
                    {"m": N * 2 * connection_factor if not disable_inst else 0, "directed": True},
                    {"m": N * connection_factor, "directed": True},
                ]

                folder_name = generate_name(
                    N,
                    num_samples,
                    graph_type,
                    N_G,
                    lag,
                    is_history_dep,
                    noise_level,
                    function_type=function_type,
                    noise_function_type=noise_function_type,
                    disable_inst=disable_inst,
                    seed=seed,
                    connection_factor=connection_factor,
                    base_noise_type=base_noise_type
                )
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
                    base_noise_type=base_noise_type
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Temporal Synthetic Data Generator")
    parser.add_argument("--config_file", type=str)
    args = parser.parse_args()

    main(config_file=args.config_file)
