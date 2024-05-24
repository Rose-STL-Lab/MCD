"""
Borrowed from github.com/microsoft/causica
"""

from typing import List, Optional, Tuple, Type, Union
from torch.nn import Dropout, LayerNorm, Linear, Module, Sequential
import torch

class resBlock(Module):
    """
    Wraps an nn.Module, adding a skip connection to it.
    """

    def __init__(self, block: Module):
        """
        Args:
            block: module to which skip connection will be added. The input dimension must match the output dimension.
        """
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


def generate_fully_connected(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    non_linearity: Optional[Type[Module]],
    activation: Optional[Type[Module]],
    device: torch.device,
    p_dropout: float = 0.0,
    normalization: Optional[Type[LayerNorm]] = None,
    res_connection: bool = False,
) -> Module:
    """
    Generate a fully connected network.

    Args:
        input_dim: Int. Size of input to network.
        output_dim: Int. Size of output of network.
        hidden_dims: List of int. Sizes of internal hidden layers. i.e. [a, b] is three linear layers with shapes (input_dim, a), (a, b), (b, output_dim)
        non_linearity: Non linear activation function used between Linear layers.
        activation: Final layer activation to use.
        device: torch device to load weights to.
        p_dropout: Float. Dropout probability at the hidden layers.
        init_method: initialization method
        normalization: Normalisation layer to use (batchnorm, layer norm, etc). Will be placed before linear layers, excluding the input layer.
        res_connection : Whether to use residual connections where possible (if previous layer width matches next layer width)

    Returns:
        Sequential object containing the desired network.
    """
    layers: List[Module] = []

    prev_dim = input_dim
    for idx, hidden_dim in enumerate(hidden_dims):

        block: List[Module] = []

        if normalization is not None and idx > 0:
            block.append(normalization(prev_dim).to(device))
        block.append(Linear(prev_dim, hidden_dim).to(device))

        if non_linearity is not None:
            block.append(non_linearity())
        if p_dropout != 0:
            block.append(Dropout(p_dropout))

        if res_connection and (prev_dim == hidden_dim):
            layers.append(resBlock(Sequential(*block)))
        else:
            layers.append(Sequential(*block))
        prev_dim = hidden_dim

    if normalization is not None:
        layers.append(normalization(prev_dim).to(device))
    layers.append(Linear(prev_dim, output_dim).to(device))

    if activation is not None:
        layers.append(activation())

    fcnn = Sequential(*layers)
    return fcnn
