#!/usr/bin/env python3
"""
Simple training loop example for easymltorch.

This example demonstrates how to use the easymltorch Model class to train
an agent to match a random target action based on input states.

The synthetic environment:
- Generates random input states
- Has a fixed target action for each unique state (simulated via random mapping)
- Rewards correct action predictions positively and incorrect ones negatively
"""

import random

import numpy as np

from easymltorch import Model


def main() -> None:
    """Run the simple training loop example."""
    # Configuration
    input_size = 10
    hidden_size = 32
    num_actions = 4
    num_episodes = 1000
    learning_rate = 0.01

    # Create the model
    print("Creating model...")
    model = Model(
        inputs=input_size,
        hidden=hidden_size,
        outputs=num_actions,
        lr=learning_rate,
        device="cpu",  # Use CPU for this example
    )

    print(f"Model created with {input_size} inputs, {hidden_size} hidden units, "
          f"{num_actions} outputs")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {model.device}")
    print()

    # Training loop
    print("Starting training...")
    print("-" * 50)

    total_reward = 0.0
    correct_count = 0
    window_size = 100

    # For reproducibility
    random.seed(42)
    np.random.seed(42)

    for episode in range(num_episodes):
        # Generate a random state
        state = np.random.randn(input_size).astype(np.float32)

        # Determine the "correct" action for this state
        # We use a simple hash-based approach to create a consistent target
        state_hash = hash(tuple(np.round(state, 2).tolist()))
        target_action = abs(state_hash) % num_actions

        # Get the model's prediction
        predicted_action = model.predict(state)

        # Determine reward based on whether prediction matches target
        if predicted_action == target_action:
            reward = 1.0
            correct_count += 1
        else:
            reward = -1.0

        # Apply the reward
        model.reward(reward)
        total_reward += reward

        # Print progress every 100 episodes
        if (episode + 1) % window_size == 0:
            accuracy = correct_count / window_size * 100
            avg_reward = total_reward / window_size
            print(
                f"Episode {episode + 1:4d}: "
                f"Accuracy: {accuracy:5.1f}% | "
                f"Avg Reward: {avg_reward:+.2f}"
            )
            # Reset counters for next window
            total_reward = 0.0
            correct_count = 0

    print("-" * 50)
    print("Training complete!")
    print()

    # Demonstration: Run a few predictions
    print("Demonstration predictions:")
    print("-" * 50)

    for i in range(5):
        state = np.random.randn(input_size).astype(np.float32)
        state_hash = hash(tuple(np.round(state, 2).tolist()))
        target_action = abs(state_hash) % num_actions

        predicted_action = model.predict(state)
        match = "CORRECT" if predicted_action == target_action else "wrong"

        print(f"  State {i + 1}: Predicted={predicted_action}, "
              f"Target={target_action} [{match}]")

        # Reset state since we don't want to train here
        model.reset()

    print()

    # Save the trained model
    save_path = "trained_model.pt"
    print(f"Saving model to '{save_path}'...")
    model.save(save_path)
    print("Model saved successfully!")


if __name__ == "__main__":
    main()
