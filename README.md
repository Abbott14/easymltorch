# easymltorch

A lightweight reinforcement-learning-style wrapper around PyTorch that simplifies model creation, prediction, and reward-based training while keeping full PyTorch power accessible.

## Features

- Simple, intuitive API for reward-based learning
- Built-in flexible neural network architecture (`SimpleNet`)
- Policy gradient training out of the box
- Automatic device selection (CUDA/CPU)
- Full PyTorch compatibility for custom networks
- Easy model saving and loading

## Installation

### From source (development)

```bash
git clone https://github.com/Abbott14/easymltorch.git
cd easymltorch
pip install -e .
```

### Dependencies

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.20.0

## Quick Start

```python
from easymltorch import Model

# Create a model with 10 inputs, 32 hidden units, and 2 output actions
model = Model(10, 32, 2)

# Get a state from your environment
state = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Make a prediction (returns an integer action)
action = model.predict(state)

# Apply a reward based on the action's outcome
if action == expected_action:
    model.reward(1.0)   # Positive reward for correct action
else:
    model.reward(-1.0)  # Negative reward for incorrect action
```

## API Reference

### Model

The main class for creating and training models.

```python
Model(
    inputs: int,           # Number of input features
    hidden: int | list,    # Hidden layer size(s)
    outputs: int,          # Number of output actions
    lr: float = 1e-3,      # Learning rate
    device: str = None,    # Device ('cpu', 'cuda', or auto-detect)
    network: nn.Module = None,  # Optional custom network
    strategy: str = "policy"    # Reward strategy
)
```

#### Methods

- **`predict(x)`**: Make a prediction and return an integer action
  - Accepts: `list`, `numpy.ndarray`, or `torch.Tensor`
  - Returns: `int` (action index)

- **`reward(r)`**: Apply a reward and update the model
  - Accepts: `int` or `float` (reward value)
  - Positive rewards reinforce the last action
  - Negative rewards discourage the last action

- **`reset()`**: Clear stored prediction state without applying reward

- **`save(path)`**: Save model and optimizer state to file

- **`load(path)`**: Load model and optimizer state from file

### SimpleNet

A flexible fully-connected neural network included with the package.

```python
from easymltorch import SimpleNet

# Single hidden layer
net = SimpleNet(inputs=10, hidden=32, outputs=2)

# Multiple hidden layers
net = SimpleNet(inputs=10, hidden=[64, 32, 16], outputs=2)
```

## Reward Strategy

easymltorch uses the **policy gradient** (REINFORCE) algorithm for learning:

```
loss = -(reward * log_probability_of_action)
```

This means:
- **Positive rewards** increase the probability of the selected action
- **Negative rewards** decrease the probability of the selected action
- The magnitude of the reward affects how much the probability changes

The policy gradient approach is well-suited for:
- Discrete action spaces
- Environments with delayed or sparse rewards
- Learning from trial and error

## Example Training Loop

Here's a complete example of training a model to match target actions:

```python
import numpy as np
from easymltorch import Model

# Create model
model = Model(inputs=10, hidden=32, outputs=4, lr=0.01)

# Training loop
for episode in range(1000):
    # Generate random state
    state = np.random.randn(10)

    # Determine target action (your environment logic here)
    target_action = episode % 4

    # Get prediction
    predicted_action = model.predict(state)

    # Calculate and apply reward
    if predicted_action == target_action:
        model.reward(1.0)
    else:
        model.reward(-1.0)

    # Print progress
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}")

# Save trained model
model.save("my_model.pt")
```

## Using Custom Networks

You can provide your own PyTorch network:

```python
import torch.nn as nn
from easymltorch import Model

# Define custom network
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.layers(x)

# Use custom network with Model
custom_net = MyNetwork()
model = Model(
    inputs=10,
    hidden=32,  # Ignored when network is provided
    outputs=2,
    network=custom_net
)
```

## Examples

See the `examples/` folder for complete working examples:

- [`simple_training_loop.py`](examples/simple_training_loop.py) - Basic training loop with synthetic environment

Run an example:

```bash
cd examples
python simple_training_loop.py
```

## Development

### Running Tests

```bash
pip install -e ".[dev]"
pytest
```

### Code Style

This project uses Black for formatting:

```bash
black easymltorch/
```

## Architecture

The package is designed for extensibility:

```
easymltorch/
    __init__.py          # Package exports
    model.py             # Main Model class
    network.py           # Neural network architectures
    strategies/          # Reward strategies (extensible)
        policy.py        # Policy gradient implementation
    utils/               # Utility functions
        device.py        # Device handling
```

Future extensions can include:
- Additional reward strategies (Q-learning, Actor-Critic)
- Experience replay buffers
- Callbacks and hooks
- More network architectures

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
