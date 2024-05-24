import lightning.pytorch as pl
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from src.modules.CausalDecoder import CausalDecoder
import numpy as np
from src.utils.loss_utils import dag_penalty_notears, temporal_graph_sparsity
from src.utils.data_utils.data_format_utils import to_time_aggregated_graph_np, to_time_aggregated_scores_torch, zero_out_diag_np, zero_out_diag_torch
from src.utils.metrics_utils import compute_shd, get_off_diagonal
from src.model.BaseTrainer import BaseTrainer
from src.modules.adjacency_matrices.TemporalAdjacencyMatrix import TemporalAdjacencyMatrix
from src.modules.adjacency_matrices.TwoWayTemporalAdjacencyMatrix import TwoWayTemporalAdjacencyMatrix
from sklearn.metrics import f1_score, roc_auc_score

from src.training.auglag import AugLagLRConfig, AuglagLRCallback, AugLagLR, AugLagLossCalculator

class RhinoTrainer(BaseTrainer):
    """

    """

    def __init__(self,
                 full_dataset: np.array,
                 adj_matrices: np.array,
                 data_dim: int,
                 lag: int,
                 num_nodes: int,
                 causal_decoder,
                 tcsf,
                 disable_inst: bool = False,
                 likelihood_loss: str = 'flow', 
                 sparsity_factor: float = 20, 
                 num_workers: int = 16,
                 batch_size: int = 256, 
                 matrix_temperature: float = 0.25,
                 aggregated_graph: bool = False,
                 ignore_self_connections: bool = False,
                 threeway_graph_dist: bool = True,
                 skip_auglag_epochs: int = 0,
                 training_procedure: str = 'auglag',
                 training_config = None,
                 init_logits = [0, 0],
                 use_all_for_val = False,
                 shuffle = True):

        self.aggregated_graph = aggregated_graph
        self.ignore_self_connections = ignore_self_connections
        self.skip_auglag_epochs = skip_auglag_epochs
        self.init_logits = init_logits
        self.disable_inst = disable_inst
        super().__init__(full_dataset=full_dataset,
                         adj_matrices=adj_matrices,
                         data_dim=data_dim,
                         lag=lag,
                         num_workers=num_workers,
                         batch_size=batch_size,
                         aggregated_graph=self.aggregated_graph,
                         use_all_for_val=use_all_for_val,
                         shuffle=shuffle)
        self.num_nodes = num_nodes
        self.matrix_temperature = matrix_temperature
        self.threeway_graph_dist = threeway_graph_dist
        self.training_procedure = training_procedure

        print("Number of fragments:", self.num_fragments)
        print("Number of samples:", self.total_samples)
        
        assert likelihood_loss == 'mse' or likelihood_loss == 'flow'
        self.likelihood_loss = likelihood_loss

        self.sparsity_factor = sparsity_factor
        self.graphs = []

        self.causal_decoder = causal_decoder
        self.tcsf = tcsf

        self.initialize_graph()

        if self.training_procedure == 'auglag':
            self.training_config = training_config
            if training_config is None:
                self.training_config = AugLagLRConfig()
            if self.skip_auglag_epochs > 0:
                print(f"Not performing augmented lagrangian optimization for the first {self.skip_auglag_epochs} epochs...")
                self.disabled_epochs = range(self.skip_auglag_epochs)
            else:
                self.disabled_epochs = None    
            self.lr_scheduler = AugLagLR(config=self.training_config)
            self.loss_calc = AugLagLossCalculator(init_alpha=self.training_config.init_alpha,
                                                        init_rho=self.training_config.init_rho)
        
    def initialize_graph(self):
        if self.threeway_graph_dist:
            self.adj_matrix = TemporalAdjacencyMatrix(
                input_dim=self.num_nodes,
                lag=self.lag,
                tau_gumbel=self.matrix_temperature,
                init_logits=self.init_logits,
                disable_inst=self.disable_inst)
        else:
            self.adj_matrix = TwoWayTemporalAdjacencyMatrix(input_dim=self.num_nodes,
                                                            lag=self.lag,
                                                            tau_gumbel=self.matrix_temperature,
                                                            init_logits=self.init_logits,
                                                            disable_inst=self.disable_inst)
    def forward(self):
        raise NotImplementedError

    def compute_loss_terms(self, X_history: torch.Tensor, X_current: torch.Tensor, G: torch.Tensor):

        #******************* graph prior *********************
        graph_sparsity_term = self.sparsity_factor * \
            temporal_graph_sparsity(G)

        # dagness factors
        if self.training_procedure == 'auglag':
            dagness_penalty = dag_penalty_notears(G[0])

        prior_term = graph_sparsity_term
        prior_term /= self.num_fragments

        # **************** graph entropy **********************

        entropy_term = -self.adj_matrix.entropy()/self.num_fragments

        # ************* likelihood loss ***********************

        batch_size = X_history.shape[0]
        expanded_G = G.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        X_input = torch.cat((X_history, X_current.unsqueeze(1)), dim=1)
        X_pred = self.causal_decoder(X_input, expanded_G)

        mse_loss = self.compute_mse(X_current, X_pred)
        mape_loss = self.compute_mape(X_current, X_pred)

        if self.likelihood_loss == 'mse':
            likelihood_term = mse_loss

        elif self.likelihood_loss == 'flow':
            batch, num_nodes, data_dim = X_current.shape
            log_prob = self.tcsf.log_prob(X_input=(X_current - X_pred).view(batch, num_nodes*data_dim),
                                          X_history=X_history, 
                                          A=expanded_G).sum(-1)
            likelihood_term = -torch.mean(log_prob)
        
        loss_terms = {
            'graph_sparsity': graph_sparsity_term,
            'dagness_penalty': dagness_penalty,
            'graph_prior': prior_term,

            'graph_entropy': entropy_term,

            'mse': mse_loss,
            'mape': mape_loss,
            'likelihood': likelihood_term
        }
        return loss_terms
    
    def compute_loss(self, X_history, X_current, idx):
        # sample G

        G = self.adj_matrix.sample_A()
        loss_terms = self.compute_loss_terms(
            X_history=X_history,
            X_current=X_current,
            G=G)

        total_loss = loss_terms['likelihood'] +\
                     loss_terms['graph_prior'] +\
                     loss_terms['graph_entropy']
                        
        return total_loss, loss_terms, None

    def training_step(self, batch, batch_idx):
    
        X_history, X_current, adj_matrix, idx, _ = batch
        loss, loss_terms, _ = self.compute_loss(X_history, X_current, idx)
        
        loss = self.loss_calc(loss, loss_terms['dagness_penalty']/self.num_fragments)
        self.log_dict(loss_terms, on_epoch=True)

        loss_terms['loss'] = loss            
        return loss_terms

    
    def validation_func(self, X_history, X_current, adj_matrix, G, idx):
        batch_size = X_history.shape[0]

        loss, loss_terms, _ = self.compute_loss(X_history, X_current, idx)

        G = G.detach().cpu().numpy()
        adj_matrix = adj_matrix.detach().cpu().numpy()

        if self.aggregated_graph:
            G = to_time_aggregated_graph_np(G)
            if self.ignore_self_connections:
                G = zero_out_diag_np(G)

            shd_loss = compute_shd(adj_matrix, G, aggregated_graph=True)
            shd_loss = torch.Tensor([shd_loss])
            f1 = f1_score(adj_matrix.flatten(), G.flatten())
        else:
            mask = adj_matrix != G
            
            shd_loss = np.sum(mask)/batch_size
            shd_inst = np.sum(mask[:, 0])/batch_size
            shd_lag = np.sum(mask[:, 1:])/batch_size
            
            # shd_loss, shd_inst, shd_lag = compute_shd(adj_matrix, G)
            tp = np.logical_and(adj_matrix == 1, adj_matrix == G)
            fp = np.logical_and(adj_matrix != 1, G == 1)
            fn = np.logical_and(adj_matrix != 0, G == 0)
            
            f1_inst = 2*np.sum(tp[:, 0]) / (2*np.sum(tp[:, 0]) + np.sum(fp[:, 0]) + np.sum(fn[:, 0]))
            f1_lag = 2*np.sum(tp[:, 1:]) / (2*np.sum(tp[:, 1:]) + np.sum(fp[:, 1:]) + np.sum(fn[:, 1:]))
            
            # f1_inst = f1_score(get_off_diagonal(adj_matrix[:, 0]).flatten(), get_off_diagonal(G[:, 0]).flatten())
            # f1_lag = f1_score(adj_matrix[:, 1:].flatten(), G[:, 1:].flatten())
            shd_loss = torch.Tensor([shd_loss])
            shd_inst = torch.Tensor([shd_inst])
            shd_lag = torch.Tensor([shd_lag])
             
        self.true_graph = adj_matrix[0]

        if not self.aggregated_graph:
            loss_terms['shd_inst'] = shd_inst
            loss_terms['shd_lag'] = shd_lag
            loss_terms['f1_lag'] = f1_lag
            loss_terms['f1_inst'] = f1_inst
        else:
            loss_terms['f1'] = f1

        loss_terms['shd_loss'] = shd_loss

        loss_terms['val_loss'] = loss

        for key in loss_terms:
            self.log(key, loss_terms[key])

        return loss_terms

    def validation_step(self, batch, batch_idx):
        X_history, X_current, adj_matrix, idx, _ = batch

        batch_size = X_history.shape[0]
        G = self.adj_matrix.sample_A()
        expanded_G = G.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        loss_terms = self.validation_func(X_history, X_current, adj_matrix, expanded_G, idx)     
        self.true_graph = adj_matrix[0]
        
        probs = self.adj_matrix.get_adj_matrix(do_round=False)
        probs = probs.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if self.aggregated_graph:
            probs = to_time_aggregated_scores_torch(probs)
            if self.ignore_self_connections:
                probs = zero_out_diag_torch(probs)
        
        
        return loss_terms

    def configure_optimizers(self):
        """Set the learning rates for different sets of parameters."""
        modules = {
            "functional_relationships": self.causal_decoder,
            "vardist": self.adj_matrix,
            "noise_dist": self.tcsf,
        }

        parameter_list = [
            {
                "params": module.parameters(),
                "lr": self.training_config.lr_init_dict[name],
                "name": name,
            }
            for name, module in modules.items() if module is not None
        ]

        # Check that all modules are added to the parameter list
        check_modules = set(modules.values())
        for module in self.parameters(recurse=False):
            assert module in check_modules, f"Module {module} not in module list"

        return torch.optim.Adam(parameter_list)
    
    def configure_callbacks(self):
        """Create a callback for the auglag callback."""
        if self.training_procedure == 'auglag':
            return [AuglagLRCallback(self.lr_scheduler, log_auglag=True, disabled_epochs=self.disabled_epochs)]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        X_full, adj_matrix, _ = batch
        batch_size = X_full.shape[0]

        probs = self.adj_matrix.get_adj_matrix(do_round=False).unsqueeze(0).repeat(batch_size, 1, 1, 1)

        if self.aggregated_graph:
            probs = to_time_aggregated_scores_torch(probs)
            if self.ignore_self_connections:
                probs = zero_out_diag_torch(probs)
        # G = torch.bernoulli(probs)
        G = (probs >= 0.5).long()
        return G, probs, adj_matrix

