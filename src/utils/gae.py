"""
Generalized Advantage Estimation (GAE)
Responsible: SV2 (GAE, Môi trường & Tiền xử lý)
Milestone deadline: 18/11 (Review & test: 20/11)
"""
from typing import Tuple
import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Generalized Advantage Estimation (GAE-Lambda) (T4 18/11 - SV2).

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
    A_t = delta_t + gamma * lambda * (1 - done_{t+1}) * A_{t+1}
    Returns = A_t + V(s_t)

    Args:
        rewards: Tensor of shape (n_steps,)
        values: Tensor of shape (n_steps,)
        dones: Tensor of shape (n_steps,)
        next_value: Scalar/Tensor for V(s_{n_steps})
        next_done: Done flag for step n_steps
        gamma: Discount factor (e.g. 0.99 vs 0.9 in SV2 ablation experiment)
        gae_lambda: GAE smoothing parameter (default 0.95)

    Returns:
        advantages: Tensor of shape (n_steps,)
        returns: Tensor of shape (n_steps,)
    """
    n_steps = len(rewards)
    advantages = torch.zeros(n_steps, dtype=torch.float32, device=rewards.device)
    last_gae_lam = 0.0

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_non_terminal = 1.0 - next_done.float()
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1].float()
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns
