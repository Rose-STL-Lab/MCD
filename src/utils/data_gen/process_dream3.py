import argparse
import os

import numpy as np
import torch
import pandas as pd


def process_ts(ts, timepoints, N_subjects):
    N_nodes = ts.shape[1]
    X = np.zeros((N_subjects, timepoints, N_nodes))

    for i in range(N_subjects):
        X[i] = ts[i*timepoints: (i+1)*timepoints]

    return X


def process_adj_matrix(net, size):
    A = np.zeros((size, size))
    for a, b, c in net.values:
        src = int(a[1:])-1
        dest = int(b[1:])-1
        A[src, dest] = 1
    return A


def split_by_trajectory(X, A, T=21):
    time_len = X.shape[0]
    N = X.shape[1]
    num_samples = time_len // T
    data = np.zeros((num_samples, T, N))
    adj_matrix = np.zeros((num_samples, N, N))
    for i in range(num_samples):
        data[i] = X[i*T:(i+1)*T]
        adj_matrix[i] = A

    return data, adj_matrix


def process_dream3(args):

    for size in [10, 50, 100]:
        X1 = torch.load(os.path.join(
            args.dataset_dir, 'Dream3TensorData', f'Size{size}Ecoli1.pt'))['TsData'].numpy()
        A1 = pd.read_table(os.path.join(
            args.dataset_dir, 'TrueGeneNetworks', f'InSilicoSize{size}-Ecoli1.tsv'), header=None)
        A1 = process_adj_matrix(A1, size)
        X1, A1 = split_by_trajectory(X1, A1)

        X2 = torch.load(os.path.join(
            args.dataset_dir, 'Dream3TensorData', f'Size{size}Ecoli2.pt'))['TsData'].numpy()
        A2 = pd.read_table(os.path.join(
            args.dataset_dir, 'TrueGeneNetworks', f'InSilicoSize{size}-Ecoli2.tsv'), header=None)
        A2 = process_adj_matrix(A2, size)
        X2, A2 = split_by_trajectory(X2, A2)

        if not os.path.exists(os.path.join(args.save_dir, f'ecoli_{size}', 'grouped_by_matrix')):
            os.makedirs(os.path.join(args.save_dir,
                        f'ecoli_{size}', 'grouped_by_matrix'))

        np.savez(os.path.join(
            args.save_dir, f'ecoli_{size}', 'grouped_by_matrix', 'ecoli_1.npz'), X=X1, adj_matrix=A1)
        np.savez(os.path.join(
            args.save_dir, f'ecoli_{size}', 'grouped_by_matrix', 'ecoli_2.npz'), X=X2, adj_matrix=A2)

        X = np.concatenate((X1, X2), axis=0)
        A = np.concatenate((A1, A2), axis=0)
        np.savez(os.path.join(args.save_dir,
                 f'ecoli_{size}', 'ecoli.npz'), X=X, adj_matrix=A)

        X11 = torch.load(os.path.join(
            args.dataset_dir, 'Dream3TensorData', f'Size{size}Yeast1.pt'))['TsData'].numpy()
        A11 = pd.read_table(os.path.join(
            args.dataset_dir, 'TrueGeneNetworks', f'InSilicoSize{size}-Yeast1.tsv'), header=None)
        A11 = process_adj_matrix(A11, size)
        X11, A11 = split_by_trajectory(X11, A11)

        X21 = torch.load(os.path.join(
            args.dataset_dir, 'Dream3TensorData', f'Size{size}Yeast2.pt'))['TsData'].numpy()
        A21 = pd.read_table(os.path.join(
            args.dataset_dir, 'TrueGeneNetworks', f'InSilicoSize{size}-Yeast2.tsv'), header=None)
        A21 = process_adj_matrix(A21, size)
        X21, A21 = split_by_trajectory(X21, A21)

        X31 = torch.load(os.path.join(
            args.dataset_dir, 'Dream3TensorData', f'Size{size}Yeast3.pt'))['TsData'].numpy()
        A31 = pd.read_table(os.path.join(
            args.dataset_dir, 'TrueGeneNetworks', f'InSilicoSize{size}-Yeast3.tsv'), header=None)
        A31 = process_adj_matrix(A31, size)
        X31, A31 = split_by_trajectory(X31, A31)

        if not os.path.exists(os.path.join(args.save_dir, f'yeast_{size}', 'grouped_by_matrix')):
            os.makedirs(os.path.join(args.save_dir,
                        f'yeast_{size}', 'grouped_by_matrix'))

        np.savez(os.path.join(
            args.save_dir, f'yeast_{size}', 'grouped_by_matrix', 'yeast_1.npz'), X=X11, adj_matrix=A11)
        np.savez(os.path.join(
            args.save_dir, f'yeast_{size}', 'grouped_by_matrix', 'yeast_2.npz'), X=X21, adj_matrix=A21)
        np.savez(os.path.join(
            args.save_dir, f'yeast_{size}', 'grouped_by_matrix', 'yeast_3.npz'), X=X31, adj_matrix=A31)
        X = np.concatenate((X11, X21, X31), axis=0)
        A = np.concatenate((A11, A21, A31), axis=0)
        print(X.shape)
        print(A.shape)
        np.savez(os.path.join(args.save_dir,
                 f'yeast_{size}', 'yeast.npz'), X=X, adj_matrix=A)

        # save combined
        X = np.concatenate((X1, X2, X11, X21, X31), axis=0)
        A = np.concatenate((A1, A2, A11, A21, A31), axis=0)
        print(X.shape)
        print(A.shape)
        np.savez(os.path.join(args.save_dir,
                 f'combined_{size}.npz'), X=X, adj_matrix=A)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    process_dream3(args)
