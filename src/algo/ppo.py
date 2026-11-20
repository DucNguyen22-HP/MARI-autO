"""
PPO Loss & Update Implementation
Responsible: SV1 (Thuật toán PPO & Huấn luyện chuyển giao)
Milestone deadline: 18/11 - 19/11 (Review & test: 20/11)
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn


def clipped_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
) -> torch.Tensor:
    """
    Computes PPO clipped surrogate objective policy loss (T4 18/11 - SV1).

    L^{CLIP}(theta) = \hat{E}_t [ min( r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]
    Returns: scalar loss (negative for gradient ascent / minimization in PyTorch).
    """
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    return policy_loss


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor = None,
    clip_ratio: float = None,
) -> torch.Tensor:
    """
    Computes value function loss with optional value clipping (T4 18/11 - SV1).

    Args:
        values: Current state value predictions V(s).
        returns: Target returns (e.g., GAE returns = advantages + old_values).
        old_values: Value predictions from previous policy (optional).
        clip_ratio: Clipping parameter for value loss (optional).
    """
    if clip_ratio is not None and old_values is not None:
        v_clipped = old_values + torch.clamp(values - old_values, -clip_ratio, clip_ratio)
        v_loss1 = (values - returns).pow(2)
        v_loss2 = (v_clipped - returns).pow(2)
        v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
    else:
        v_loss = 0.5 * (values - returns).pow(2).mean()
    return v_loss


def approx_kl_and_clipfrac(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    clip_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes approximate KL divergence and clipping fraction for diagnostics (T5 19/11 - SV1).
    Used for 500k-step health checks:
        approx_kl = mean(old_log_probs - log_probs) or 0.5 * mean((log_probs - old_log_probs)^2)
        clip_fraction = mean(abs(ratio - 1) > clip_ratio)
    """
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    # Schulman approx KL: mean((ratio - 1) - log_ratio) or simpler ratio divergence
    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clip_frac = ((ratio - 1.0).abs() > clip_ratio).float().mean()
    return approx_kl, clip_frac


def ppo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    entropies: torch.Tensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Aggregates full PPO loss: L_clip + c1 * L_vf - c2 * S[pi] (T5 19/11 - SV1).
    """
    p_loss = clipped_policy_loss(log_probs, old_log_probs, advantages, clip_ratio)
    v_loss = value_loss(values, returns)
    entropy_loss = -entropies.mean()

    total_loss = p_loss + value_coef * v_loss + entropy_coef * entropy_loss
    approx_kl, clip_frac = approx_kl_and_clipfrac(log_probs, old_log_probs, clip_ratio)

    metrics = {
        "policy_loss": p_loss.item(),
        "value_loss": v_loss.item(),
        "entropy": -entropy_loss.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_frac.item(),
        "total_loss": total_loss.item(),
    }
    return total_loss, metrics


def ppo_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout_data: Dict[str, torch.Tensor],
    batch_size: int = 64,
    n_epochs: int = 4,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> Dict[str, float]:
    """
    Runs multi-epoch minibatch PPO update (T5 19/11 - SV1).
    """
    states = rollout_data["states"]
    actions = rollout_data["actions"]
    old_log_probs = rollout_data["old_log_probs"]
    returns = rollout_data["returns"]
    advantages = rollout_data["advantages"]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = states.shape[0]
    indices = torch.randperm(dataset_size)

    accumulated_metrics: Dict[str, float] = {}

    for _ in range(n_epochs):
        for start_idx in range(0, dataset_size, batch_size):
            end_idx = min(start_idx + batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]

            b_states = states[batch_indices]
            b_actions = actions[batch_indices]
            b_old_log_probs = old_log_probs[batch_indices]
            b_returns = returns[batch_indices]
            b_advantages = advantages[batch_indices]

            log_probs, values, entropies = model.evaluate_actions(b_states, b_actions)

            loss, metrics = ppo_loss(
                log_probs=log_probs,
                old_log_probs=b_old_log_probs,
                values=values,
                returns=b_returns,
                advantages=b_advantages,
                entropies=entropies,
                clip_ratio=clip_ratio,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            for k, v in metrics.items():
                accumulated_metrics[k] = accumulated_metrics.get(k, 0.0) + v

    num_updates = n_epochs * ((dataset_size + batch_size - 1) // batch_size)
    return {k: v / max(num_updates, 1) for k, v in accumulated_metrics.items()}
