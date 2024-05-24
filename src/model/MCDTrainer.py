import lightning.pytorch as pl
import torch.nn as nn
import torch
import torch.nn.functional as F

from src.modules.MultiEmbedding import MultiEmbedding
from src.model.RhinoTrainer import RhinoTrainer
import numpy as np
from src.utils.metrics_utils import cluster_accuracy
from src.utils.loss_utils import dag_penalty_notears, temporal_graph_sparsity
from src.utils.data_utils.data_format_utils import to_time_aggregated_scores_torch, zero_out_diag_torch, get_adj_matrix_id
from src.modules.adjacency_matrices.MultiTemporalAdjacencyMatrix import MultiTemporalAdjacencyMatrix
from src.modules.MixtureSelectionLogits import MixtureSelectionLogits
from sklearn.metrics import accuracy_score, roc_auc_score
import math
from src.training.auglag import AugLagLRConfig, AugLagLR, AugLagLossCalculator

class MCDTrainer(RhinoTrainer):
    """

    """

    def __init__(self,
                 full_dataset: np.array,
                 adj_matrices: np.array,
                 data_dim: int,
                 lag: int,
                 num_nodes: int,
                 num_graphs: int,
                 causal_decoder,
                 tcsf,
                 
                 use_correct_mixture_index: bool = False, # diagnostic option to check if 
                 # the graph can be learnt when the correct mixture index is given
                 use_true_graph: bool = False, # diagnostic option to check if
                 # the correct mixture index is learnt when the correct graph is given
                 likelihood_loss: str = 'flow',
                 ignore_self_connections: bool = False,
                 sparsity_factor: float = 20,
                 num_workers: int = 16,
                 batch_size: int = 256,
                 matrix_temperature: float = 1,
                 aggregated_graph: bool = False,
                 threeway_graph_dist: bool = True,
                 skip_auglag_epochs: int = 0,
                 training_procedure: str = 'auglag',
                 training_config = None,
                 init_logits = [0, 0],
                 disable_inst = False,

                 graph_selection_prior_lambda: float = 0.0,
                 use_indices = None,
                 log_num_unique_graphs=False,
                 use_all_for_val=False,
                 shuffle=True
                 ):

        self.num_graphs = num_graphs
        self.graph_selection_prior_lambda = graph_selection_prior_lambda
        self.log_num_unique_graphs = log_num_unique_graphs
        self.init_logits = init_logits
        super().__init__(full_dataset=full_dataset,
                         adj_matrices=adj_matrices,
                         data_dim=data_dim,
                         lag=lag,
                         num_nodes=num_nodes,
                         causal_decoder=causal_decoder,
                         tcsf=tcsf,
                         likelihood_loss=likelihood_loss,
                         sparsity_factor=sparsity_factor,
                         matrix_temperature=matrix_temperature,
                         aggregated_graph=aggregated_graph,
                         threeway_graph_dist=threeway_graph_dist,
                         ignore_self_connections=ignore_self_connections,
                         use_all_for_val=use_all_for_val,
                         shuffle=shuffle,

                         num_workers=num_workers,
                         batch_size=batch_size,
                         skip_auglag_epochs=skip_auglag_epochs,
                         training_config=training_config,
                         training_procedure=training_procedure,
                         disable_inst=disable_inst
                         )
        
        # initialize point-wise logits
        self.mixture_selection = MixtureSelectionLogits(num_graphs=self.num_graphs,
                                                        num_samples=self.total_samples)

        self.use_correct_mixture_index = use_correct_mixture_index
        self.use_true_graph = use_true_graph

        # if use correct graph is set, have the set of correct graphs ready
        if self.use_true_graph:
            self.true_graphs = torch.Tensor(np.unique(adj_matrices, axis=0)) 
        
        self.use_indices = use_indices

    def initialize_graph(self):
        self.graphs = MultiTemporalAdjacencyMatrix(
            num_nodes=self.num_nodes,
            lag=self.lag,
            num_graphs=self.num_graphs,
            tau_gumbel=self.matrix_temperature,
            threeway=self.threeway_graph_dist,
            init_logits=self.init_logits,
            disable_inst=self.disable_inst
        )
    
    def set_mixture_indices(self, indices):
        self.use_indices = indices

    def forward(self):
        raise NotImplementedError
    
    def compute_loss(self, X_history, X_current, idx):
        # first, get the mixture assignment probabilities for each point
        mixture_probs = self.mixture_selection.get_probs(idx)
        # mixture_probs shape: (num_graphs, batch)

        if self.use_true_graph:
            G = self.true_graphs.to(self.device)
        else:
            # next, sample G from variational distributions
            G = self.graphs.sample_A()
        # G shape: (num_graphs, lag+1, num_nodes, num_nodes)

        
        loss_terms, component_wise_likelihood = self.compute_loss_terms(
            X_history=X_history,
            X_current=X_current,
            G=G,
            sample_idx=idx,
            mixture_probs=mixture_probs)
        
        total_loss = loss_terms['likelihood'] + \
                     loss_terms['graph_prior'] + \
                     loss_terms['graph_entropy'] + \
                     loss_terms['graph_selection_entropy'] + \
                     loss_terms['graph_selection_prior']
        
        return total_loss, loss_terms, component_wise_likelihood
    
    def compute_loss_terms(self, 
                           X_history: torch.Tensor, 
                           X_current: torch.Tensor, 
                           G: torch.Tensor,
                           sample_idx: torch.Tensor,
                           mixture_probs: torch.Tensor,
                           ):
        batch, num_nodes, data_dim = X_current.shape
        # *********** graph prior *****************
        
        # sparsity term
        graph_sparsity_term = self.sparsity_factor * \
            temporal_graph_sparsity(G)

        # dagness factors
        dagness_penalty = dag_penalty_notears(G[:, 0]) 
        graph_prior_term = graph_sparsity_term
        graph_prior_term /= self.num_fragments

        # ************* graph entropy term ************

        graph_entropy_term = -self.graphs.entropy()/self.num_fragments
        
        # ********* graph selection prior term ********

        # weight term
        ce_loss = nn.CrossEntropyLoss()
        W = -(self.graph_selection_prior_lambda * torch.arange(self.num_graphs, device=self.device)).float()
        graph_selection_prior_term = ce_loss(W.unsqueeze(1).repeat(1, batch).transpose(0, 1), self.mixture_selection.get_probs(sample_idx).transpose(0, 1))/(self.num_fragments)*self.total_samples
        # graph_selection_prior_term /= batch

        # ********** graph selection logits entropy ***

        graph_selection_entropy_term = - self.mixture_selection.entropy(sample_idx)/(self.num_fragments)*self.total_samples

        # ************* likelihood loss ****************

        X_input = torch.cat((X_history, X_current.unsqueeze(1)), dim=1)
        X_pred = self.causal_decoder(X_input, G)
        # X_pred shape: (batch, num_graphs, num_nodes, data_dim)
        
        if self.likelihood_loss == 'mse':
            X_current = X_current.unsqueeze(1).expand((-1, self.num_graphs, -1, -1))
            component_wise_likelihood =  torch.sum(torch.square(X_pred - X_current), dim=(2, 3))
            likelihood_term = mixture_probs.transpose(0, -1) * component_wise_likelihood 
            likelihood_term = torch.sum(torch.mean(likelihood_term, dim=0))

        elif self.likelihood_loss == 'flow':
            X_current = X_current.unsqueeze(1).expand((-1, self.num_graphs, -1, -1))
            log_prob = self.tcsf.log_prob(X_input=(X_current - X_pred).reshape(batch*self.num_graphs, -1),
                                          X_history=X_history, 
                                          A=G).view(batch, self.num_graphs, -1).sum(-1)

            # weight the likelihood term by the mixture selection probabilities
            # mixture_probs.shape: (num_graphs, batch), log_prob.shape: (batch, self.num_graphs)
            component_wise_likelihood = log_prob * mixture_probs.transpose(0, -1)
            likelihood_term = -torch.sum(torch.mean(component_wise_likelihood, dim=0))
            
        mixture_index = torch.argmax(mixture_probs, dim=0)
        mse_loss = self.compute_mse(X_current[:, mixture_index], X_pred[:, mixture_index])
        mape_loss = self.compute_mape(X_current[:, mixture_index], X_pred[:, mixture_index])
        
        # ************************************************
        loss_terms = {
            'graph_sparsity': graph_sparsity_term,
            'dagness_penalty': dagness_penalty,
            'graph_prior': graph_prior_term,

            'graph_entropy': graph_entropy_term,
            'graph_selection_entropy': graph_selection_entropy_term,
            'graph_selection_prior': graph_selection_prior_term,

            'mse': mse_loss,
            'mape': mape_loss,
            'likelihood': likelihood_term
        }
        return loss_terms, component_wise_likelihood
   
    def training_step(self, batch, batch_idx):

        X_history, X_current, adj_matrix, idx, mixture_idx = batch
        
        if self.use_correct_mixture_index:
            self.mixture_selection.manual_set_mixture_indices(idx, mixture_idx)

        if self.use_indices != None:
            self.use_indices = self.use_indices.to(self.device).long()
            self.mixture_selection.manual_set_mixture_indices(idx, self.use_indices[idx])
        
        loss, loss_terms, _ = self.compute_loss(X_history, X_current, idx)

        self.track_gradients(self.mixture_selection, "mixture_grad")
    
        if self.log_num_unique_graphs:
            graph_index = torch.argmax(self.mixture_selection.get_probs(idx), dim=0)
            self.log('num_unique_graphs', float(torch.unique(graph_index).shape[0]))
        loss_terms['loss'] = loss
        return loss_terms

    # convert nan gradients to zero
    def on_after_backward(self) -> None:
        for p in self.parameters():
            if p.grad != None:
                p.grad = torch.nan_to_num(p.grad)
        return super().on_after_backward()
    
    def update_validation_indices(self, X_history, X_current, idx):
        with torch.inference_mode(False):
            with torch.enable_grad():
                optimizer = self.optimizers()
                
                for param in self.parameters():
                    param.requires_grad_(False)

                self.mixture_selection.requires_grad_(True)
                loss, _, _ = self.compute_loss(X_history=X_history,
                                        X_current=X_current,
                                        idx=idx)
                optimizer.zero_grad()
                loss.backward()        
                optimizer.step()
                
                for param in self.parameters():
                    param.requires_grad_(True)

    def validation_step(self, batch, batch_idx):
        X_history, X_current, adj_matrix, idx, mixture_idx = batch

        # use the correct offset
        if not self.use_all_for_val:
            idx = idx + self.train_dataset_np.shape[0]

        if self.use_correct_mixture_index:
            self.mixture_selection.manual_set_mixture_indices(idx, mixture_idx)
        elif not self.use_all_for_val:
            # update the mixture indices for the validation dataset
            self.update_validation_indices(X_history, X_current, idx)
            
        # first select graph for each sample
        graph_index = torch.argmax(self.mixture_selection.get_probs(idx), dim=0)
        if self.use_true_graph:
            G = self.true_graphs.to(self.device)[graph_index]
        else:
            G = self.graphs.sample_A()[graph_index]
        
        loss_terms = self.validation_func(X_history, X_current, adj_matrix, G, idx)

        # if we are using the correct graph, evaluate the accuracy with which
        # the correct mixture index is being learnt
        if self.use_true_graph:
            pred_mixture_idx = self.mixture_selection.get_mixture_indices(idx).detach().cpu().numpy()
            true_idx = mixture_idx.detach().cpu().numpy()
            mixture_idx_acc = accuracy_score(true_idx, pred_mixture_idx)
            loss_terms['mixture_idx_acc'] = mixture_idx_acc
            self.log('mixture_idx_acc', mixture_idx_acc)
                
        if not (self.use_true_graph or self.use_correct_mixture_index):
            # log the cluster accuracy
            true_idx = mixture_idx.detach().cpu().numpy()
            graph_idx = graph_index.detach().cpu().numpy()
            loss_terms['cluster_acc'] = cluster_accuracy(true_idx, graph_idx)
            self.log('cluster_acc', loss_terms['cluster_acc'])

        # first select graph for each sample
        graph_index = torch.argmax(self.mixture_selection.get_probs(idx), dim=0)    
        # get the corresponding graph probabilities
        probs = self.graphs.get_adj_matrix(do_round=False)[graph_index]
        if self.aggregated_graph:
            probs = to_time_aggregated_scores_torch(probs)
            if self.ignore_self_connections:
                probs = zero_out_diag_torch(probs)

        return loss_terms


    def configure_optimizers(self):
        """Set the learning rates for different sets of parameters."""
        modules = {
            "functional_relationships": self.causal_decoder,
            "vardist": self.graphs,
            "noise_dist": self.tcsf,
            "mixing_probs": self.mixture_selection
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        X_full, adj_matrix, idx = batch
        batch_size = X_full.shape[0]

        # first select graph for each sample
        graph_index = torch.argmax(self.mixture_selection.get_probs(idx), dim=0)    
        # get the corresponding graph probabilities
        probs = self.graphs.get_adj_matrix(do_round=False)[graph_index]
        if self.aggregated_graph:
            probs = to_time_aggregated_scores_torch(probs)
            if self.ignore_self_connections:
                probs = zero_out_diag_torch(probs)
        # G = torch.bernoulli(probs)
        G = (probs >= 0.5).long()
        return G, probs, adj_matrix

    def get_cluster_indices(self):
        true_cluster_idx = get_adj_matrix_id(self.adj_matrices_np)[1]
        pred_cluster_idx = self.mixture_selection.get_mixture_indices(torch.arange(self.total_samples))

        return true_cluster_idx, pred_cluster_idx
