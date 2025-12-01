# Binary Classifier Example

A complete example demonstrating how to use `easymltorch` to train a model that classifies inputs as "positive sum" or "negative sum".

## The Task

Given a vector of numbers, the model learns to predict:
- **Action 0**: The sum of inputs is negative
- **Action 1**: The sum of inputs is positive or zero

## Files

- `train.py` - Train the model
- `evaluate.py` - Evaluate a trained model
- `environment.py` - Simple environment generating training data

## Quick Start

```bash
# Train the model (creates model.pt)
python train.py

# Evaluate the trained model
python evaluate.py
```

## Expected Output

After training, you should see accuracy improving over episodes:

```
Episode  100: Accuracy: 52.0% | Avg Reward: +0.04
Episode  200: Accuracy: 58.0% | Avg Reward: +0.16
Episode  300: Accuracy: 67.0% | Avg Reward: +0.34
...
Episode 1000: Accuracy: 95.0% | Avg Reward: +0.90
```

The evaluation script will show how well the model generalizes to new data.
