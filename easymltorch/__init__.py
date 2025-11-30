"""
easymltorch - A lightweight reinforcement-learning-style wrapper around PyTorch.

This package provides a simplified interface for machine-learning agents that
learn from reward signals, while keeping full PyTorch power accessible.

Example:
    >>> from easymltorch import Model
    >>> model = Model(10, 32, 2)
    >>> action = model.predict(state)
    >>> model.reward(1.0 if action == expected else -1.0)
"""

from .model import Model
from .network import SimpleNet

__version__ = "0.1.0"
__all__ = ["Model", "SimpleNet"]
