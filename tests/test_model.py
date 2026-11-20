"""
Unit Tests for ActorCritic Neural Network Model
Responsible: SV3 (Mốc kiểm thử: T6 20/11)
Tests:
- ActorCritic.__init__
- forward pass output shapes
- get_action_and_value output shapes and types
"""
import pytest
import torch
from src.models.actor_critic import ActorCritic


def test_actor_critic_shapes():
    batch_size = 4
    num_actions = 7
    model = ActorCritic(in_channels=1, num_actions=num_actions)
    dummy_input = torch.randn(batch_size, 1, 84, 84)

    logits, values = model(dummy_input)

    assert logits.shape == (batch_size, num_actions)
    assert values.shape == (batch_size, 1)


def test_actor_critic_sampling():
    model = ActorCritic(in_channels=1, num_actions=7)
    dummy_input = torch.randn(2, 1, 84, 84)

    action, log_prob, entropy, value = model.get_action_and_value(dummy_input)

    assert action.shape == (2,)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2,)
    assert (action >= 0).all() and (action < 7).all()


def test_evaluate_actions():
    model = ActorCritic(in_channels=1, num_actions=7)
    dummy_input = torch.randn(3, 1, 84, 84)
    dummy_actions = torch.tensor([0, 2, 6])

    log_prob, value, entropy = model.evaluate_actions(dummy_input, dummy_actions)

    assert log_prob.shape == (3,)
    assert value.shape == (3,)
    assert entropy.shape == (3,)
