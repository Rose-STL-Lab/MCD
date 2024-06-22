import lightning.pytorch as pl
from torch import nn
import torch
import torch.nn.functional as F
import torch.distributions as td


class MixtureSelectionLogits(pl.LightningModule):

    def __init__(
        self,
        num_samples: int,
        num_graphs: int,
        tau: float = 1.0
    ):

        super().__init__()
        self.num_graphs = num_graphs
        self.num_samples = num_samples
        self.graph_select_logits = nn.Parameter((
            torch.ones(self.num_graphs, self.num_samples,
                       device=self.device) * 0.01
        ), requires_grad=True)
        self.tau = tau

    def manual_set_mixture_indices(self, idx, mixture_idx):
        """
        Use this function to manually set the mixture index.
        Mainly used for diagnostic/ablative purposes
        """

        with torch.no_grad():
            self.graph_select_logits[:, idx] = -10
            self.graph_select_logits[mixture_idx, idx] = 10
        self.graph_select_logits.requires_grad_(False)

    def set_logits(self, idx, logits):
        """
        Use this function to manually set the logits.
        Used in the E step of the EM implementation
        """
        with torch.no_grad():
            self.graph_select_logits[:, idx] = logits.transpose(0, -1)

    def reset_parameters(self):
        with torch.no_grad():
            self.graph_select_logits[:] = torch.ones(self.num_graphs,
                                                     self.num_samples,
                                                     device=self.device) * 0.01

    def turn_off_grad(self):
        self.graph_select_logits.requires_grad_(False)

    def turn_on_grad(self):
        self.graph_select_logits.requires_grad_(True)

    def get_probs(self, idx):
        return F.softmax(self.graph_select_logits[:, idx]/self.tau, dim=0)

    def get_mixture_indices(self, idx):
        return torch.argmax(self.graph_select_logits[:, idx], dim=0)

    def entropy(self, idx):
        logits = self.graph_select_logits[:, idx]/self.tau
        dist = td.Categorical(logits=logits.transpose(0, -1))
        entropy = dist.entropy().sum()

        return entropy/idx.shape[0]
