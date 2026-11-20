"""
Unit Tests for PPO Loss & Update
Responsible: SV1 (Mốc kiểm thử: T6 20/11)
Tests:
- clipped_policy_loss
- value_loss
- approx_kl_and_clipfrac
- ppo_loss
- ppo_update
"""
import pytest
import torch
import torch.nn as nn
from src.algo.ppo import (
    clipped_policy_loss,
    value_loss,
    approx_kl_and_clipfrac,
    ppo_loss,
    ppo_update,
)
from src.models.actor_critic import ActorCritic


def test_clipped_policy_loss(sample_tensors):
    loss = clipped_policy_loss(
        log_probs=sample_tensors["log_probs"],
        old_log_probs=sample_tensors["old_log_probs"],
        advantages=sample_tensors["advantages"],
        clip_ratio=0.2,
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    # Check backward gradient
    loss.backward()
    assert sample_tensors["log_probs"].grad is not None


def test_value_loss(sample_tensors):
    v_loss = value_loss(
        values=sample_tensors["values"],
        returns=sample_tensors["returns"],
    )
    assert isinstance(v_loss, torch.Tensor)
    assert v_loss.dim() == 0
    assert v_loss.item() >= 0.0


def test_approx_kl_and_clipfrac(sample_tensors):
    kl, clip_frac = approx_kl_and_clipfrac(
        log_probs=sample_tensors["log_probs"],
        old_log_probs=sample_tensors["old_log_probs"],
        clip_ratio=0.2,
    )
    assert isinstance(kl, torch.Tensor)
    assert isinstance(clip_frac, torch.Tensor)
    assert kl.item() >= 0.0
    assert 0.0 <= clip_frac.item() <= 1.0


def test_ppo_loss(sample_tensors):
    loss, metrics = ppo_loss(
        log_probs=sample_tensors["log_probs"],
        old_log_probs=sample_tensors["old_log_probs"],
        values=sample_tensors["values"],
        returns=sample_tensors["returns"],
        advantages=sample_tensors["advantages"],
        entropies=sample_tensors["entropies"],
    )
    assert isinstance(loss, torch.Tensor)
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "approx_kl" in metrics
    assert "clip_fraction" in metrics


def test_ppo_update():
    model = ActorCritic(in_channels=1, num_actions=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rollout_size = 32

    rollout_data = {
        "states": torch.randn(rollout_size, 1, 84, 84),
        "actions": torch.randint(0, 7, (rollout_size,)),
        "old_log_probs": torch.randn(rollout_size),
        "returns": torch.randn(rollout_size),
        "advantages": torch.randn(rollout_size),
    }

    metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        rollout_data=rollout_data,
        batch_size=8,
        n_epochs=2,
    )
    assert "policy_loss" in metrics
    assert "approx_kl" in metrics
