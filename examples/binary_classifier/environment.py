"""
Simple environment for binary classification.

The task is to predict whether the sum of input features is positive or negative.
"""

import numpy as np


class BinaryClassifierEnv:
    """
    Environment that generates random vectors and rewards correct classification.

    The model must learn to predict:
    - Action 0: sum of inputs < 0 (negative)
    - Action 1: sum of inputs >= 0 (positive or zero)
    """

    def __init__(self, input_size: int = 10, seed: int | None = None) -> None:
        """
        Initialize the environment.

        Args:
            input_size: Number of input features.
            seed: Random seed for reproducibility.
        """
        self.input_size = input_size
        self.rng = np.random.default_rng(seed)
        self._current_state: np.ndarray | None = None
        self._correct_action: int | None = None

    def reset(self) -> np.ndarray:
        """
        Generate a new random state.

        Returns:
            A numpy array of shape (input_size,) with values in [-1, 1].
        """
        # Generate random values between -1 and 1
        self._current_state = self.rng.uniform(-1, 1, size=self.input_size)
        self._correct_action = 1 if self._current_state.sum() >= 0 else 0
        return self._current_state.astype(np.float32)

    def get_correct_action(self) -> int:
        """
        Get the correct action for the current state.

        Returns:
            0 if sum is negative, 1 if sum is positive or zero.
        """
        if self._correct_action is None:
            raise RuntimeError("Call reset() first to generate a state")
        return self._correct_action

    def step(self, action: int) -> tuple[float, bool]:
        """
        Evaluate the action and return reward.

        Args:
            action: The predicted action (0 or 1).

        Returns:
            Tuple of (reward, is_correct).
        """
        if self._correct_action is None:
            raise RuntimeError("Call reset() first to generate a state")

        is_correct = action == self._correct_action

        # Reward: +1 for correct, -1 for incorrect
        reward = 1.0 if is_correct else -1.0

        return reward, is_correct
