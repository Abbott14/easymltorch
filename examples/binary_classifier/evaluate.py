#!/usr/bin/env python3
"""
Evaluate a trained binary classifier model.

This script demonstrates:
- Loading a saved model
- Running evaluation without training (using reset() to skip reward)
- Computing and displaying test metrics
"""

import argparse
from pathlib import Path

from easymltorch import Model
from environment import BinaryClassifierEnv


def evaluate(
    model_path: str = "model.pt",
    input_size: int = 10,
    hidden_size: int = 32,
    num_tests: int = 500,
    seed: int = 123,  # Different seed for test data
) -> None:
    """
    Evaluate the trained model on test data.

    Args:
        model_path: Path to the saved model.
        input_size: Size of input vector (must match training).
        hidden_size: Size of hidden layer (must match training).
        num_tests: Number of test samples.
        seed: Random seed for test data.
    """
    print("=" * 60)
    print("Binary Classifier Evaluation")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Test samples: {num_tests}")
    print("=" * 60)
    print()

    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found.")
        print("Please run train.py first to create a trained model.")
        return

    # Create model with same architecture
    model = Model(
        inputs=input_size,
        hidden=hidden_size,
        outputs=2,
        device="cpu",
    )

    # Load trained weights
    model.load(model_path)
    print(f"Loaded model from: {Path(model_path).absolute()}")
    print()

    # Create test environment with different seed
    env = BinaryClassifierEnv(input_size=input_size, seed=seed)

    # Evaluation metrics
    correct = 0
    predictions = {0: 0, 1: 0}
    correct_per_class = {0: 0, 1: 0}
    total_per_class = {0: 0, 1: 0}

    print("Evaluating...")
    print("-" * 60)

    for i in range(num_tests):
        # Get test sample
        state = env.reset()
        correct_action = env.get_correct_action()

        # Make prediction (no training)
        predicted_action = model.predict(state)

        # Reset model state (skip reward step)
        model.reset()

        # Track metrics
        predictions[predicted_action] += 1
        total_per_class[correct_action] += 1

        if predicted_action == correct_action:
            correct += 1
            correct_per_class[correct_action] += 1

        # Show some example predictions
        if i < 5:
            result = "CORRECT" if predicted_action == correct_action else "WRONG"
            print(
                f"  Sample {i+1}: "
                f"Sum={state.sum():+.2f} -> "
                f"Predicted={predicted_action}, "
                f"Actual={correct_action} [{result}]"
            )

    if num_tests > 5:
        print("  ...")

    print("-" * 60)
    print()

    # Display results
    accuracy = correct / num_tests * 100
    print("Results:")
    print(f"  Overall Accuracy: {accuracy:.1f}% ({correct}/{num_tests})")
    print()

    print("Per-class accuracy:")
    for cls in [0, 1]:
        if total_per_class[cls] > 0:
            cls_acc = correct_per_class[cls] / total_per_class[cls] * 100
            label = "Negative" if cls == 0 else "Positive"
            print(
                f"  {label} (class {cls}): "
                f"{cls_acc:.1f}% ({correct_per_class[cls]}/{total_per_class[cls]})"
            )

    print()
    print("Prediction distribution:")
    print(f"  Predicted Negative (0): {predictions[0]} times")
    print(f"  Predicted Positive (1): {predictions[1]} times")


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained binary classifier"
    )
    parser.add_argument(
        "--model-path", type=str, default="model.pt", help="Path to saved model"
    )
    parser.add_argument(
        "--input-size", type=int, default=10, help="Size of input vector"
    )
    parser.add_argument(
        "--hidden-size", type=int, default=32, help="Size of hidden layer"
    )
    parser.add_argument(
        "--num-tests", type=int, default=500, help="Number of test samples"
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for test data"
    )

    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_tests=args.num_tests,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
