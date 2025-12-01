"""Basic tests for the Model class."""

import copy

import numpy as np
import pytest
import torch

from easymltorch import Model


class TestModelInstantiation:
    """Tests for Model instantiation."""

    def test_create_model_with_int_hidden(self) -> None:
        """Test creating a model with a single hidden layer size."""
        model = Model(inputs=10, hidden=32, outputs=2)
        assert model.net is not None
        assert model.optim is not None
        assert model.device is not None

    def test_create_model_with_list_hidden(self) -> None:
        """Test creating a model with multiple hidden layers."""
        model = Model(inputs=10, hidden=[64, 32, 16], outputs=4)
        assert model.net is not None

    def test_create_model_with_custom_lr(self) -> None:
        """Test creating a model with custom learning rate."""
        model = Model(inputs=10, hidden=32, outputs=2, lr=0.01)
        # Check that optimizer has the correct learning rate
        for param_group in model.optim.param_groups:
            assert param_group["lr"] == 0.01

    def test_create_model_with_cpu_device(self) -> None:
        """Test creating a model on CPU."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        assert model.device == torch.device("cpu")

    def test_create_model_with_custom_network(self) -> None:
        """Test creating a model with a custom network."""
        custom_net = torch.nn.Sequential(
            torch.nn.Linear(10, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
        model = Model(inputs=10, hidden=32, outputs=2, network=custom_net)
        assert model.net is custom_net

    def test_invalid_strategy_raises_error(self) -> None:
        """Test that an invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            Model(inputs=10, hidden=32, outputs=2, strategy="qlearning")


class TestModelPredict:
    """Tests for Model.predict() method."""

    def test_predict_returns_int(self) -> None:
        """Test that predict returns an integer."""
        model = Model(inputs=10, hidden=32, outputs=4, device="cpu")
        x = [0.1] * 10
        action = model.predict(x)
        assert isinstance(action, int)

    def test_predict_with_list_input(self) -> None:
        """Test predict with Python list input."""
        model = Model(inputs=5, hidden=16, outputs=3, device="cpu")
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        action = model.predict(x)
        assert 0 <= action < 3

    def test_predict_with_numpy_input(self) -> None:
        """Test predict with numpy array input."""
        model = Model(inputs=5, hidden=16, outputs=3, device="cpu")
        x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        action = model.predict(x)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_predict_with_torch_input(self) -> None:
        """Test predict with torch tensor input."""
        model = Model(inputs=5, hidden=16, outputs=3, device="cpu")
        x = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        action = model.predict(x)
        assert isinstance(action, int)
        assert 0 <= action < 3

    def test_predict_stores_state(self) -> None:
        """Test that predict stores internal state."""
        model = Model(inputs=5, hidden=16, outputs=3, device="cpu")
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        action = model.predict(x)
        assert model._last_input is not None
        assert model._last_logits is not None
        assert model._last_action == action

    def test_predict_with_random_vector(self) -> None:
        """Test predict with random vector of correct size."""
        model = Model(inputs=20, hidden=64, outputs=5, device="cpu")
        x = np.random.randn(20)
        action = model.predict(x)
        assert isinstance(action, int)
        assert 0 <= action < 5


class TestModelReward:
    """Tests for Model.reward() method."""

    def test_reward_without_predict_raises_error(self) -> None:
        """Test that reward without prior predict raises RuntimeError."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        with pytest.raises(RuntimeError, match="without a prior predict"):
            model.reward(1.0)

    def test_reward_after_predict_succeeds(self) -> None:
        """Test that reward after predict completes without error."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        x = [0.1] * 10
        model.predict(x)
        # Should not raise
        model.reward(1.0)

    def test_reward_clears_state(self) -> None:
        """Test that reward clears internal state."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        x = [0.1] * 10
        model.predict(x)
        model.reward(1.0)
        assert model._last_input is None
        assert model._last_logits is None
        assert model._last_action is None

    def test_double_reward_raises_error(self) -> None:
        """Test that calling reward twice without predict raises error."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        x = [0.1] * 10
        model.predict(x)
        model.reward(1.0)
        with pytest.raises(RuntimeError, match="without a prior predict"):
            model.reward(1.0)

    def test_parameters_change_after_reward(self) -> None:
        """Test that model parameters change after reward is applied."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")

        # Get initial parameter values
        initial_params = [p.clone() for p in model.net.parameters()]

        # Perform prediction and reward
        x = [0.1] * 10
        model.predict(x)
        model.reward(10.0)  # Large reward to ensure noticeable change

        # Check that at least some parameters have changed
        params_changed = False
        for initial, current in zip(initial_params, model.net.parameters()):
            if not torch.allclose(initial, current):
                params_changed = True
                break

        assert params_changed, "Parameters should change after reward"

    def test_reward_with_negative_value(self) -> None:
        """Test that negative reward works correctly."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        x = [0.1] * 10
        model.predict(x)
        # Should not raise
        model.reward(-5.0)


class TestModelReset:
    """Tests for Model.reset() method."""

    def test_reset_clears_state(self) -> None:
        """Test that reset clears internal state."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        x = [0.1] * 10
        model.predict(x)
        model.reset()
        assert model._last_input is None
        assert model._last_logits is None
        assert model._last_action is None

    def test_reset_allows_new_predict(self) -> None:
        """Test that reset allows a new predict without reward."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        x = [0.1] * 10
        model.predict(x)
        model.reset()
        # Should be able to predict again
        action = model.predict(x)
        assert isinstance(action, int)


class TestModelSaveLoad:
    """Tests for Model.save() and load() methods."""

    def test_save_and_load(self, tmp_path) -> None:
        """Test saving and loading model."""
        model = Model(inputs=10, hidden=32, outputs=2, device="cpu")

        # Get initial parameters
        initial_params = {
            name: p.clone() for name, p in model.net.named_parameters()
        }

        # Make some predictions and rewards to change the model
        for _ in range(5):
            x = [0.1] * 10
            model.predict(x)
            model.reward(1.0)

        # Save the model
        save_path = tmp_path / "model.pt"
        model.save(str(save_path))

        # Create a new model and load
        new_model = Model(inputs=10, hidden=32, outputs=2, device="cpu")
        new_model.load(str(save_path))

        # Check that parameters match
        for name, p in new_model.net.named_parameters():
            assert torch.allclose(
                p, dict(model.net.named_parameters())[name]
            ), f"Parameter {name} does not match after load"
