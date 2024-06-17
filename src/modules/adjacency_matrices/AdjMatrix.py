from abc import ABC, abstractmethod
import torch


class AdjMatrix(ABC):
    """
    Adjacency matrix interface for DECI
    """

    @abstractmethod
    def get_adj_matrix(self, do_round: bool = True) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        raise NotImplementedError()

    @abstractmethod
    def entropy(self) -> torch.Tensor:
        """
        Computes the entropy of distribution q. In this case 0.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample_A(self) -> torch.Tensor:
        """
        Returns the adjacency matrix.
        """
        raise NotImplementedError()
