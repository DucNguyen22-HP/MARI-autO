"""
Rollout Buffer for On-Policy Experience Collection
Used by: SV2, SV3 and SV1 for collecting transitions during PPO steps.
"""
from typing import Dict
import torch


class RolloutBuffer:
    """
    Stores on-policy trajectories collected across environment steps.
    """
    def __init__(self, n_steps: int, obs_shape: tuple, device: torch.device):
        self.n_steps = n_steps
        self.obs_shape = obs_shape
        self.device = device

        self.states = torch.zeros((n_steps, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(n_steps, dtype=torch.long, device=device)
        self.rewards = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.dones = torch.zeros(n_steps, dtype=torch.bool, device=device)
        self.values = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(n_steps, dtype=torch.float32, device=device)

        self.advantages = torch.zeros(n_steps, dtype=torch.float32, device=device)
        self.returns = torch.zeros(n_steps, dtype=torch.float32, device=device)

        self.step = 0

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        done: bool,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ):
        if self.step >= self.n_steps:
            raise IndexError("RolloutBuffer is full.")

        self.states[self.step] = state
        self.actions[self.step] = action
        self.rewards[self.step] = float(reward)
        self.dones[self.step] = bool(done)
        self.values[self.step] = value.squeeze()
        self.log_probs[self.step] = log_prob.squeeze()
        self.step += 1

    def compute_returns_and_advantages(self, compute_gae_fn, next_value, next_done, gamma=0.99, gae_lambda=0.95):
        advs, rets = compute_gae_fn(
            rewards=self.rewards,
            values=self.values,
            dones=self.dones,
            next_value=next_value,
            next_done=next_done,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        self.advantages = advs
        self.returns = rets

    def get_data(self) -> Dict[str, torch.Tensor]:
        return {
            "states": self.states,
            "actions": self.actions,
            "old_log_probs": self.log_probs,
            "returns": self.returns,
            "advantages": self.advantages,
            "values": self.values,
        }

    def reset(self):
        self.step = 0
