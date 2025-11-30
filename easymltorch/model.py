"""Main Model class for easymltorch."""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch
import torch.nn as nn

from .network import SimpleNet
from .strategies import policy as policy_strategy
from .utils.device import get_device


class Model:
    """
    A high-level wrapper for reinforcement-learning-style training with PyTorch.

    This class provides a simplified interface for creating models that learn
    from reward signals using policy gradient methods.

    Args:
        inputs: Number of input features.
        hidden: Size of hidden layer(s). Can be an int for a single hidden
                layer or a list of ints for multiple hidden layers.
        outputs: Number of output actions/classes.
        lr: Learning rate for the Adam optimizer. Defaults to 1e-3.
        device: Device to use ('cpu', 'cuda', etc.). If None, automatically
                selects 'cuda' if available, otherwise 'cpu'.
        network: Optional custom nn.Module to use instead of SimpleNet.
                 Must accept inputs and produce outputs of the specified sizes.
        strategy: The reward strategy to use. Currently only 'policy' is
                  supported. Defaults to 'policy'.

    Attributes:
        net: The underlying PyTorch neural network.
        optim: The Adam optimizer.
        device: The torch device being used.

    Example:
        >>> model = Model(10, 32, 2)
        >>> action = model.predict([0.1, 0.2, ...])  # Returns int
        >>> model.reward(1.0)  # Positive reward for good action
    """

    def __init__(
        self,
        inputs: int,
        hidden: Union[int, list[int]],
        outputs: int,
        lr: float = 1e-3,
        device: str | None = None,
        network: nn.Module | None = None,
        strategy: str = "policy",
    ) -> None:
        self.device = get_device(device)

        # Create or use provided network
        if network is not None:
            self.net = network.to(self.device)
        else:
            self.net = SimpleNet(inputs, hidden, outputs).to(self.device)

        # Set up optimizer
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Store strategy
        if strategy != "policy":
            raise ValueError(
                f"Unknown strategy '{strategy}'. Currently only 'policy' is supported."
            )
        self._strategy = strategy
        self._strategy_module = policy_strategy

        # Internal state tracking
        self._last_input: torch.Tensor | None = None
        self._last_logits: torch.Tensor | None = None
        self._last_action: int | None = None

    def predict(self, x: Union[list[Any], np.ndarray, torch.Tensor]) -> int:
        """
        Make a prediction (forward pass) and return the selected action.

        The input is processed through the network, and the action with
        the highest logit value is selected (argmax). The input and logits
        are stored for the subsequent reward() call.

        Args:
            x: Input state. Can be a Python list, numpy array, or torch tensor.
               Shape should be (inputs,) or (1, inputs).

        Returns:
            The selected action as an integer (0 to outputs-1).

        Note:
            This method does NOT compute or apply gradients. Gradients are
            only computed when reward() is called.
        """
        # Convert input to tensor
        if isinstance(x, list):
            tensor_input = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif isinstance(x, np.ndarray):
            tensor_input = torch.from_numpy(x).float().to(self.device)
        elif isinstance(x, torch.Tensor):
            tensor_input = x.float().to(self.device)
        else:
            raise TypeError(
                f"Input must be list, numpy array, or torch tensor, got {type(x)}"
            )

        # Ensure input is 1D (unbatched) for storage
        if tensor_input.dim() == 2 and tensor_input.shape[0] == 1:
            tensor_input = tensor_input.squeeze(0)

        # Forward pass (no gradient tracking needed here)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(tensor_input)

        # Store for reward computation (need to recompute with gradients later)
        self._last_input = tensor_input
        self._last_logits = logits

        # Select action (argmax)
        action = int(torch.argmax(logits).item())
        self._last_action = action

        return action

    def reward(self, r: Union[int, float]) -> None:
        """
        Apply a reward signal and update the model using policy gradient.

        This method computes the policy gradient loss based on the last
        prediction and the provided reward, then performs a gradient
        descent step.

        Args:
            r: The reward signal. Positive values reinforce the last action,
               negative values discourage it.

        Raises:
            RuntimeError: If called before predict() or if predict() has not
                         been called since the last reward().

        Note:
            After calling this method, the stored prediction state is cleared.
            You must call predict() again before the next reward().
        """
        if self._last_input is None or self._last_action is None:
            raise RuntimeError("reward() called without a prior predict() step")

        reward = float(r)

        # Re-run forward pass with gradient tracking
        self.net.train()
        logits = self.net(self._last_input)

        # Temporarily store logits for strategy to access
        stored_logits = self._last_logits
        self._last_logits = logits

        # Compute loss using the strategy
        loss = self._strategy_module.compute_loss(self, reward)

        # Restore and then clear state
        self._last_logits = stored_logits

        # Perform gradient descent
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Clear stored context
        self.reset()

    def reset(self) -> None:
        """
        Clear the stored prediction state.

        This is called automatically after reward(), but can also be called
        manually to discard a prediction without applying a reward.
        """
        self._last_input = None
        self._last_logits = None
        self._last_action = None

    def save(self, path: str) -> None:
        """
        Save the model and optimizer state to a file.

        Args:
            path: Path to save the checkpoint to.

        Example:
            >>> model.save("model_checkpoint.pt")
        """
        checkpoint = {
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        """
        Load the model and optimizer state from a file.

        Args:
            path: Path to load the checkpoint from.

        Example:
            >>> model.load("model_checkpoint.pt")
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
