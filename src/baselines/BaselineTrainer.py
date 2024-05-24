
from typing import Any
import lightning.pytorch as pl
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from src.dataset.BaselineTSDataset import BaselineTSDataset
import numpy as np
from src.utils.metrics_utils import mape_loss

class BaselineTrainer(pl.LightningModule):
    # trainer class for the baselines
    # they do not need training
    def __init__(self,
                 full_dataset: np.array,
                 adj_matrices: np.array,
                 data_dim: int,
                 num_nodes: int,
                 lag: int,
                 num_workers: int = 16,
                 aggregated_graph: bool = False):
        
        super().__init__()

        self.num_workers = num_workers
        self.aggregated_graph = aggregated_graph
        self.data_dim = data_dim
        self.lag = lag
        self.num_nodes = num_nodes
        self.full_dataset_np = full_dataset
        self.adj_matrices_np = adj_matrices
        self.total_samples = full_dataset.shape[0]
        assert adj_matrices.shape[0] == self.total_samples

        self.full_dataset = BaselineTSDataset(
            X = self.full_dataset_np, 
            adj_matrix = self.adj_matrices_np, 
            lag=lag, 
            aggregated_graph=self.aggregated_graph,
            return_graph_indices=True
            )

        self.batch_size = 1

    def forward(self):
        raise NotImplementedError

    def compute_mse(self, X_current, X_pred):
        return F.mse_loss(X_current, X_pred)

    def compute_mape(self, X_current, X_pred):
        return mape_loss(X_current, X_pred)

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def get_full_dataloader(self) -> DataLoader:
        return DataLoader(self.full_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        raise NotImplementedError