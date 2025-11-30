"""Neural network architectures for easymltorch."""

from typing import Union

import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """
    A simple fully-connected neural network with ReLU activations.

    This network can be configured with a single hidden layer size (int)
    or multiple hidden layers (list of ints).

    Args:
        inputs: Number of input features.
        hidden: Size of hidden layer(s). Can be an int for a single hidden
                layer or a list of ints for multiple hidden layers.
        outputs: Number of output features.

    Example:
        >>> # Single hidden layer with 32 units
        >>> net = SimpleNet(10, 32, 2)
        >>> # Multiple hidden layers: 64 -> 32 -> 16
        >>> net = SimpleNet(10, [64, 32, 16], 2)
    """

    def __init__(
        self,
        inputs: int,
        hidden: Union[int, list[int]],
        outputs: int,
    ) -> None:
        super().__init__()

        # Normalize hidden to a list
        if isinstance(hidden, int):
            hidden_sizes = [hidden]
        else:
            hidden_sizes = list(hidden)

        if not hidden_sizes:
            raise ValueError("hidden must be a non-empty int or list of ints")

        # Build layers
        layers: list[nn.Module] = []
        prev_size = inputs

        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size

        # Final output layer (no activation)
        layers.append(nn.Linear(prev_size, outputs))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, inputs) or (inputs,).

        Returns:
            Output tensor of shape (batch_size, outputs) or (outputs,).
        """
        return self.network(x)
