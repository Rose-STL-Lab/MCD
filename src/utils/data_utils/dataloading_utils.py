import numpy as np
import os

def get_dataset_path(dataset):
    if 'netsim' in dataset:
        dataset_path = 'netsim'
    elif 'dream3' in dataset:
        dataset_path = 'dream3'
    elif 'snp100' in dataset:
        dataset_path = 'snp100'
    elif dataset == 'lorenz96' or dataset == 'finance' or dataset == 'fluxnet':
        dataset_path = dataset
    else:
        dataset_path = 'synthetic'
    
    return dataset_path

def create_save_name(dataset, cfg):
    if dataset == 'lorenz96':
        return f'lorenz96_N={cfg.num_nodes}_T={cfg.timesteps}_num_graphs={cfg.num_graphs}'
    else:
        return dataset

def load_synthetic_from_folder(dataset_dir, dataset_name):
    X = np.load(os.path.join(dataset_dir, dataset_name, 'X.npy'))
    adj_matrix = np.load(os.path.join(
        dataset_dir, dataset_name, 'adj_matrix.npy'))

    return X, adj_matrix

def load_netsim(dataset_dir, dataset_file):
    # load the files
    data = np.load(os.path.join(dataset_dir, dataset_file + '.npz'))
    X = data['X']
    adj_matrix = data['adj_matrix']
    # adj_matrix = np.transpose(adj_matrix, (0, 2, 1))
    return X, adj_matrix

def load_dream3_combined(dataset_dir, size):
    data = np.load(os.path.join(dataset_dir, f'combined_{size}.npz'))
    X = data['X']
    adj_matrix = data['adj_matrix']
    return X, adj_matrix


def load_snp100(dataset, dataset_dir):
    if dataset == 'snp100':
        X = np.load(os.path.join(dataset_dir, 'X.npy'))
    else:
        # get the sector
        sector = dataset.split('_')[1]
        X = np.load(os.path.join(dataset_dir, f'X_{sector}.npy'))
        
    D = X.shape[2]
    # we do not have the true adjacency matrix 
    adj_matrix = np.zeros((X.shape[0], D, D))
    return X, adj_matrix

def load_data(dataset, dataset_dir, config):
    if 'netsim' in dataset:
        X, adj_matrix = load_netsim(
            dataset_dir=dataset_dir, dataset_file=dataset)
        aggregated_graph = True
        # adj_matrix = np.transpose(adj_matrix, (0, 2, 1))
        # read lag from config file
        lag = int(config['lag'])
        data_dim = 1
        X = np.expand_dims(X, axis=-1)        
    elif dataset == 'dream3':
        dream3_size = int(config['dream3_size'])
        X, adj_matrix = load_dream3_combined(dataset_dir=dataset_dir, size=dream3_size)
        lag = int(config['lag'])
        data_dim = 1
        aggregated_graph=True 
        X = np.expand_dims(X, axis=-1)
    elif 'snp100' in dataset:
        X, adj_matrix = load_snp100(dataset=dataset, dataset_dir=dataset_dir)
        aggregated_graph = True
        lag = 1
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
    else:
        X, adj_matrix = load_synthetic_from_folder(
            dataset_dir=dataset_dir, dataset_name=dataset)
        lag = adj_matrix.shape[1] - 1
        data_dim = 1
        X = np.expand_dims(X, axis=-1)
        aggregated_graph = False
    print("Loaded data of shape:", X.shape)
    return X, adj_matrix, aggregated_graph, lag, data_dim