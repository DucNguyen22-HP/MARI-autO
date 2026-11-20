"""
Pytest Fixtures and Mock Environments
Used across all tests for SV1, SV2, and SV3.
"""
import pytest
import torch
import numpy as np
import gym
from gym import spaces


class MockMarioEnv(gym.Env):
    """
    Mock Gym Environment simulating Super Mario Bros for fast testing without ROMs/NES-py dependencies.
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=255, shape=(240, 256, 3), dtype=np.uint8)
        self.steps = 0
        self.x_pos = 40

    def reset(self):
        self.steps = 0
        self.x_pos = 40
        return np.random.randint(0, 255, size=(240, 256, 3), dtype=np.uint8)

    def step(self, action):
        self.steps += 1
        self.x_pos += 2 if action > 0 else 0
        done = self.steps >= 20
        reward = 1.0
        info = {
            "x_pos": self.x_pos,
            "flag_get": self.x_pos >= 70,
            "time": 400 - self.steps,
        }
        obs = np.random.randint(0, 255, size=(240, 256, 3), dtype=np.uint8)
        return obs, reward, done, info


@pytest.fixture
def mock_mario_env():
    return MockMarioEnv()


@pytest.fixture
def sample_tensors():
    torch.manual_seed(42)
    batch_size = 16
    log_probs = torch.randn(batch_size, requires_grad=True)
    old_log_probs = log_probs.detach() + torch.randn(batch_size) * 0.05
    advantages = torch.randn(batch_size)
    values = torch.randn(batch_size, requires_grad=True)
    returns = torch.randn(batch_size)
    entropies = torch.rand(batch_size)
    return {
        "log_probs": log_probs,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "values": values,
        "returns": returns,
        "entropies": entropies,
    }
