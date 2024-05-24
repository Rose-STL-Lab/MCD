
import lightning.pytorch as pl
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, ConstantLR
from src.dataset.FragmentDataset import FragmentDataset
import numpy as np
from src.utils.metrics_utils import mape_loss

class BaseTrainer(pl.LightningModule):

    def __init__(self,
                 full_dataset: np.array,
                 adj_matrices: np.array,
                 data_dim: int,
                 lag: int,
                 num_workers: int = 16,
                 batch_size: int = 256,
                 aggregated_graph: bool = False,
                 return_graph_indices: bool = True,
                 val_frac: float = 0.2,
                 use_all_for_val: bool = False,
                 shuffle: bool = True):
        super().__init__()
        # self.automatic_optimization = False
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.aggregated_graph = aggregated_graph
        self.data_dim = data_dim
        self.lag = lag
        self.return_graph_indices = return_graph_indices

        I = np.arange(full_dataset.shape[0])
        if shuffle:
            rng = np.random.default_rng()
            rng.shuffle(I)

        self.full_dataset_np = full_dataset[I]
        self.adj_matrices_np = adj_matrices[I]

        self.use_all_for_val = use_all_for_val
            
        self.val_frac = val_frac
        assert self.val_frac >= 0.0 and self.val_frac < 1.0, "Validation fraction should be between 0 and 1"

        num_samples = full_dataset.shape[0]
        self.total_samples = num_samples
        assert adj_matrices.shape[0] == num_samples

        if self.use_all_for_val:
            print("Using *all* examples for validation. Ignoring val_frac...")
            self.train_dataset_np = self.full_dataset_np
            self.val_dataset_np = self.full_dataset_np
            
            self.train_adj_np = self.adj_matrices_np
            self.val_adj_np = self.adj_matrices_np
        else:
            self.train_dataset_np = self.full_dataset_np[:int((1-self.val_frac)*num_samples)]
            self.val_dataset_np = self.full_dataset_np[int((1-self.val_frac)*num_samples):]
            
            self.train_adj_np = self.adj_matrices_np[:int((1-self.val_frac)*num_samples)]
            self.val_adj_np = self.adj_matrices_np[int((1-self.val_frac)*num_samples):]
        
        self.train_frag_dataset = FragmentDataset(
            self.train_dataset_np, 
            self.train_adj_np, 
            lag=lag, 
            aggregated_graph=self.aggregated_graph,
            return_graph_indices=self.return_graph_indices)
        
        self.val_frag_dataset = FragmentDataset(
            self.val_dataset_np, 
            self.val_adj_np, 
            lag=lag, 
            aggregated_graph=self.aggregated_graph,
            return_graph_indices=self.return_graph_indices)
        
        # self.full_frag_dataset = FragmentDataset(
        #     self.full_dataset_np, 
        #     self.adj_matrices_np, 
        #     lag=lag, 
        #     aggregated_graph=self.aggregated_graph,
        #     return_graph_indices=self.return_graph_indices)
        
        self.num_fragments = len(self.train_frag_dataset)
        
        self.full_dataset = TensorDataset(
                                torch.Tensor(self.full_dataset_np), 
                                torch.Tensor(self.adj_matrices_np),
                                torch.arange(self.full_dataset_np.shape[0]))
        
        if self.batch_size is None:
            # do full-batch training
            self.batch_size = self.num_fragments
            
    def forward(self):
        raise NotImplementedError

    def compute_loss(self, X_history, X_current, X_full, adj_matrix):
        raise NotImplementedError

    def compute_mse(self, X_current, X_pred):
        return F.mse_loss(X_current, X_pred)

    def compute_mape(self, X_current, X_pred):
        return mape_loss(X_current, X_pred)

    def training_step(self, batch, batch_idx):
        X_history, X_current, X_full, adj_matrix = batch
        loss = self.compute_loss(X_history, X_current, X_full, adj_matrix)
        return loss

    def validation_step(self, batch, batch_idx):
        X_history, X_current, X_full, adj_matrix = batch
        loss = self.compute_loss(X_history, X_current, X_full, adj_matrix)

        self.log("val_loss", loss)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_frag_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_frag_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def get_full_dataloader(self) -> DataLoader:
        return DataLoader(self.full_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def track_gradients(self, m, log_name):
        total_norm = 0
        for p in m.parameters():
            if p.grad != None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.log(log_name, total_norm)