import os
import csv
import numpy as np


def standard_scaling(X, across_samples=False):
    # expected X of shape (n_samples, timesteps, num_nodes, data_dim) or (n_samples, timesteps, num_nodes)

    if across_samples:
        means = np.mean(X, axis=(0, 1))[np.newaxis, np.newaxis, :]
        std = np.std(X, axis=(0, 1))[np.newaxis, np.newaxis, :]
    else:
        means = np.mean(X, axis=(1))[:, np.newaxis]
        std = np.std(X, axis=(1))[:, np.newaxis]

    eps = 1e-6
    Y = (X-means) / (std + eps)

    return Y


def min_max_scaling(X, across_samples=False):
    # expected X of shape (n_samples, timesteps, num_nodes, data_dim) or (n_samples, timesteps, num_nodes)

    if across_samples:
        mins = np.amin(X, axis=(0, 1))[np.newaxis, np.newaxis, :]
        maxs = np.amax(X, axis=(0, 1))[np.newaxis, np.newaxis, :]
    else:
        mins = np.amin(X, axis=(1))[:, np.newaxis]
        maxs = np.amax(X, axis=(1))[:, np.newaxis]

    Y = (X-mins) / (maxs - mins) * 2 - 1

    return Y


def write_results_to_disk(dataset, metrics):
    # write results to file
    results_dir = os.path.join('results', dataset)
    results_file = os.path.join(results_dir, 'results.csv')
    file_exists = os.path.isfile(results_file)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    with open(results_file, 'a', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows([metrics])
