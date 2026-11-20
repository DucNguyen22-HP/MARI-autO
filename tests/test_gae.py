"""
Unit Tests for Generalized Advantage Estimation (GAE)
Responsible: SV2 (Mốc kiểm thử: T6 20/11)
Tests:
- compute_gae calculation correctness and terminal boundary handling
"""
import pytest
import torch
from src.utils.gae import compute_gae


def test_compute_gae_shapes():
    n_steps = 10
    rewards = torch.ones(n_steps)
    values = torch.zeros(n_steps)
    dones = torch.zeros(n_steps, dtype=torch.bool)
    next_value = torch.tensor(0.0)
    next_done = torch.tensor(False)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        next_done=next_done,
        gamma=0.99,
        gae_lambda=0.95,
    )

    assert advantages.shape == (n_steps,)
    assert returns.shape == (n_steps,)
    assert not torch.isnan(advantages).any()
    assert not torch.isnan(returns).any()


def test_compute_gae_done_boundary():
    # If done is True at step 5, next steps should not leak advantage back across terminal
    n_steps = 6
    rewards = torch.ones(n_steps)
    values = torch.ones(n_steps)
    dones = torch.tensor([False, False, False, False, True, False])
    next_value = torch.tensor(1.0)
    next_done = torch.tensor(False)

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        next_value=next_value,
        next_done=next_done,
        gamma=0.9,
        gae_lambda=0.9,
    )

    assert len(advantages) == n_steps
    assert returns[0] == advantages[0] + values[0]
