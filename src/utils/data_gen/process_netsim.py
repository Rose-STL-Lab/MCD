import scipy.io as sio
import argparse
import os
import numpy as np
import math
import random

def process_ts(ts, timepoints, N_subjects):
    N_nodes = ts.shape[1]
    X = np.zeros((N_subjects, timepoints, N_nodes))

    for i in range(N_subjects):
        X[i] = ts[i*timepoints: (i+1)*timepoints]

    return X

def process_adj_matrix(net):
    return (np.abs(np.swapaxes(net, 1, 2)) > 0).astype(int)
    
def process_netsim(args):
    seed = 0
    simulations = range(1, 29)
    random.seed(seed)
    np.random.seed(seed)
    N_t_dict = {}
    X_dict_by_matrix = {}

    for i in simulations:
        mat = sio.loadmat(os.path.join(args.dataset_dir, f'sim{i}.mat'))
        timepoints = mat['Ntimepoints'][0, 0]
        N_subjects = mat['Nsubjects'][0, 0] 
        ts = mat['ts']
        net = mat['net']
        N_nodes = ts.shape[1]
        X = process_ts(ts, timepoints, N_subjects)
        adj_matrix = process_adj_matrix(net)

        if (N_nodes, timepoints) not in N_t_dict:
            N_t_dict[(N_nodes, timepoints)] = {}
            N_t_dict[(N_nodes, timepoints)]['X'] = []
            N_t_dict[(N_nodes, timepoints)]['adj_matrix'] = []
        
        N_t_dict[(N_nodes, timepoints)]['X'].append(X)
        N_t_dict[(N_nodes, timepoints)]['adj_matrix'].append(adj_matrix)
        
        for j in range(X.shape[0]):
            if (adj_matrix[j].tobytes(), X.shape[1]) not in X_dict_by_matrix:
                X_dict_by_matrix[(adj_matrix[j].tobytes(), X.shape[1])] = []
            X_dict_by_matrix[(adj_matrix[j].tobytes(), X.shape[1])].append(X[j])

        if N_nodes == 15 and timepoints == 200:
            print("SIMULATION", i)
    for N_nodes, timepoints in N_t_dict:
        X = np.concatenate(N_t_dict[(N_nodes, timepoints)]['X'], axis=0)
        adj_matrix = np.concatenate(N_t_dict[(N_nodes, timepoints)]['adj_matrix'], axis=0)
        np.savez(os.path.join(args.save_dir, f'netsim_{N_nodes}_{timepoints}.npz'), X=X, adj_matrix=adj_matrix)

    counts = {}
    for key, T in X_dict_by_matrix:
        N = int(math.sqrt(len(np.frombuffer(key, dtype=int))))
        if (N, T) not in counts:
            counts[(N, T)] = 0
        
        adj_matrix = np.array(np.frombuffer(key, dtype=int)).reshape(N, N)
        if not os.path.exists(os.path.join(args.save_dir, 'grouped_by_matrix', f'{N}_{T}')):
            os.makedirs(os.path.join(args.save_dir, 'grouped_by_matrix', f'{N}_{T}'))
        X = np.array(X_dict_by_matrix[(key, T)])
        np.savez(os.path.join(args.save_dir, 'grouped_by_matrix', f'{N}_{T}', f'netsim_{counts[(N, T)]}.npz'), X=X, adj_matrix=adj_matrix)
        counts[(N, T)] += 1

    # add the permutations
    for N_nodes in [15, 50]:
        timepoints = 200
        num_graphs = 3
        permutation_pool = [np.random.permutation(np.arange(N_nodes)) for i in range(num_graphs)]

        X = []
        adj_matrix = []
        
        for i in range(len(N_t_dict[(N_nodes, timepoints)]['X'][0])):
            I = random.choice(permutation_pool)
            x = N_t_dict[(N_nodes, timepoints)]['X'][0][i]
            x = np.array(x)
            x = x[:, I]

            G = N_t_dict[(N_nodes, timepoints)]['adj_matrix'][0][i]
            G = G[I][:, I]
            
            X.append(x)
            adj_matrix.append(G)

        X = np.array(X)
        adj_matrix = np.array(adj_matrix)
        np.savez(os.path.join(args.save_dir, f'netsim_{N_nodes}_{timepoints}_permuted.npz'), X=X, adj_matrix=adj_matrix)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()
    process_netsim(args)
