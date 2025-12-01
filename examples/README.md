# Example Projects

This directory contains example projects demonstrating how to use the `easymltorch` package.

## Setup

1. **Install the package** (from the repository root):

```bash
# Clone the repository
git clone https://github.com/Abbott14/easymltorch.git
cd easymltorch

# Install in development mode
pip install -e .
```

2. **Verify installation**:

```bash
python -c "from easymltorch import Model; print('Installation successful!')"
```

## Examples

### 1. Simple Training Loop (`simple_training_loop.py`)

Basic example showing how to train a model to match target actions.

```bash
cd examples
python simple_training_loop.py
```

### 2. Binary Classifier (`binary_classifier/`)

Complete project demonstrating a binary classification task where the model learns to predict whether a sum of inputs is positive or negative.

```bash
cd examples/binary_classifier
python train.py
python evaluate.py
```

## Creating Your Own Project

```python
from easymltorch import Model

# 1. Create a model
model = Model(
    inputs=10,      # Number of input features
    hidden=32,      # Hidden layer size (or list like [64, 32])
    outputs=2,      # Number of possible actions
    lr=0.01         # Learning rate
)

# 2. Training loop
for episode in range(1000):
    state = get_state_from_environment()  # Your environment
    action = model.predict(state)

    # Determine reward based on action outcome
    reward = calculate_reward(action)

    # Update the model
    model.reward(reward)

# 3. Save the trained model
model.save("my_model.pt")
```
