#!/usr/bin/env python3
"""
Train a binary classifier using easymltorch.

This script demonstrates:
- Creating a Model with custom parameters
- Running a training loop with reward-based learning
- Tracking and displaying training progress
- Saving the trained model
"""

import argparse
from pathlib import Path

from easymltorch import Model
from environment import BinaryClassifierEnv


def train(
    episodes: int = 2000,
    input_size: int = 10,
    hidden_size: int = 32,
    learning_rate: float = 0.01,
    log_interval: int = 100,
    save_path: str = "model.pt",
    seed: int = 42,
) -> None:
    """
    Train the binary classifier model.

    Args:
        episodes: Number of training episodes.
        input_size: Size of input vector.
        hidden_size: Size of hidden layer.
        learning_rate: Learning rate for optimizer.
        log_interval: Episodes between progress logs.
        save_path: Path to save the trained model.
        seed: Random seed for reproducibility.
    """
    print("=" * 60)
    print("Binary Classifier Training")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Input size: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Save path: {save_path}")
    print("=" * 60)
    print()

    # Create environment and model
    env = BinaryClassifierEnv(input_size=input_size, seed=seed)
    model = Model(
        inputs=input_size,
        hidden=hidden_size,
        outputs=2,  # Binary classification: 0 or 1
        lr=learning_rate,
        device="cpu",
    )

    print(f"Model device: {model.device}")
    print()

    # Training metrics
    correct_count = 0
    total_reward = 0.0

    print("Training...")
    print("-" * 60)

    for episode in range(1, episodes + 1):
        # Get new state from environment
        state = env.reset()

        # Make prediction
        action = model.predict(state)

        # Get reward from environment
        reward, is_correct = env.step(action)

        # Update model with reward
        model.reward(reward)

        # Track metrics
        total_reward += reward
        if is_correct:
            correct_count += 1

        # Log progress
        if episode % log_interval == 0:
            accuracy = correct_count / log_interval * 100
            avg_reward = total_reward / log_interval

            print(
                f"Episode {episode:5d}: "
                f"Accuracy: {accuracy:5.1f}% | "
                f"Avg Reward: {avg_reward:+.2f}"
            )

            # Reset metrics for next interval
            correct_count = 0
            total_reward = 0.0

    print("-" * 60)
    print("Training complete!")
    print()

    # Save the model
    model.save(save_path)
    print(f"Model saved to: {Path(save_path).absolute()}")


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train a binary classifier with easymltorch"
    )
    parser.add_argument(
        "--episodes", type=int, default=2000, help="Number of training episodes"
    )
    parser.add_argument(
        "--input-size", type=int, default=10, help="Size of input vector"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=32, help="Size of hidden layer"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate"
    )
    parser.add_argument(
        "--log-interval", type=int, default=100, help="Episodes between logs"
    )
    parser.add_argument(
        "--save-path", type=str, default="model.pt", help="Path to save model"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )

    args = parser.parse_args()

    train(
        episodes=args.episodes,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        learning_rate=args.lr,
        log_interval=args.log_interval,
        save_path=args.save_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
