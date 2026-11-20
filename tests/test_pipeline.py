"""
Integration Test for End-to-End Pipeline
Responsible: SV3 & Cả nhóm (T6 20/11: Chạy test_pipeline.py kiểm tra toàn diện trước huấn luyện 1-1)
Verifies:
- Environment wrapping -> Rollout Collection -> GAE calculation -> PPO Update -> Model evaluation
"""
import pytest
import torch
from src.env.wrappers import FrameSkip, GrayResize, ScalarizeReward
from src.models.actor_critic import ActorCritic
from src.algo.ppo import ppo_update
from src.utils.gae import compute_gae
from src.utils.buffer import RolloutBuffer


def test_pipeline_integration(mock_mario_env):
    # Wrap mock environment
    env = FrameSkip(mock_mario_env, skip=4)
    env = GrayResize(env, shape=(84, 84))
    env = ScalarizeReward(env)

    # Initialize model & buffer
    device = torch.device("cpu")
    model = ActorCritic(in_channels=1, num_actions=7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_steps = 16
    buffer = RolloutBuffer(n_steps=n_steps, obs_shape=(1, 84, 84), device=device)

    # Rollout collection
    obs = env.reset()
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    for _ in range(n_steps):
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(state)

        next_obs, reward, done, info = env.step(action.item())
        buffer.add(
            state=state.squeeze(0),
            action=action,
            reward=reward,
            done=done,
            value=value,
            log_prob=log_prob,
        )
        if done:
            obs = env.reset()
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        _, _, _, next_val = model.get_action_and_value(state)

    # GAE
    buffer.compute_returns_and_advantages(
        compute_gae_fn=compute_gae,
        next_value=next_val.squeeze(),
        next_done=torch.tensor(done, device=device),
        gamma=0.99,
        gae_lambda=0.95,
    )

    # PPO update
    metrics = ppo_update(
        model=model,
        optimizer=optimizer,
        rollout_data=buffer.get_data(),
        batch_size=8,
        n_epochs=2,
    )

    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "approx_kl" in metrics
    print(f"\n[+] Pipeline Integration Test Passed! Metrics: {metrics}")
