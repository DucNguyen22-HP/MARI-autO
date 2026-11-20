"""
Unit Tests for Environment Wrappers
Responsible: SV2 (Mốc kiểm thử: T6 20/11)
Tests:
- FrameSkip
- GrayResize
- ScalarizeReward
"""
import pytest
import numpy as np
from src.env.wrappers import FrameSkip, GrayResize, ScalarizeReward


def test_frameskip(mock_mario_env):
    env = FrameSkip(mock_mario_env, skip=4)
    obs = env.reset()
    assert obs.shape == (240, 256, 3)

    next_obs, reward, done, info = env.step(1)
    # 4 frames skipped, so reward is accumulated 4 * 1.0 = 4.0
    assert reward == 4.0
    assert not done


def test_gray_resize(mock_mario_env):
    env = GrayResize(mock_mario_env, shape=(84, 84))
    obs = env.reset()
    assert obs.shape == (1, 84, 84)
    assert obs.dtype == np.float32
    assert 0.0 <= obs.min() <= obs.max() <= 1.0


def test_scalarize_reward(mock_mario_env):
    env = ScalarizeReward(
        mock_mario_env,
        progress_weight=1.0,
        death_penalty=-10.0,
        flag_bonus=50.0,
        time_penalty=-0.1,
    )
    env.reset()
    _, reward, done, info = env.step(1)
    assert isinstance(reward, float)
