"""
Main Training Script for Super Mario Bros PPO
Used by: SV1, SV2, SV3 for their designated machines and experimental setups.
Supports 500k-step health check diagnostics (CN 22/11).
"""
import argparse
import os
import time
import yaml
import torch
import numpy as np

from src.env.make_env import make_mario_env
from src.models.actor_critic import ActorCritic
from src.algo.ppo import ppo_update
from src.utils.gae import compute_gae
from src.utils.buffer import RolloutBuffer
from src.utils.logger import MetricsLogger
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mario PPO Agent")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to checkpoint for transfer learning")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["experiment"].get("seed", 42)
    set_seed(seed)

    device_str = cfg["experiment"].get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    print(f"[*] Starting experiment: {cfg['experiment']['name']} on device: {device} with seed: {seed}")

    # Directories
    checkpoint_dir = cfg["output"].get("checkpoint_dir", "checkpoints")
    log_dir = cfg["output"].get("log_dir", "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    logger = MetricsLogger(log_dir=log_dir, filename="progress.csv")

    # Environment
    env_cfg = cfg["env"]
    env = make_mario_env(
        env_id=env_cfg["env_id"],
        action_type=env_cfg.get("action_type", "simple"),
        skip_frames=env_cfg.get("skip_frames", 4),
        img_size=tuple(env_cfg.get("img_size", [84, 84])),
        shaped_reward=env_cfg.get("shaped_reward", True),
        reward_kwargs=env_cfg.get("reward_kwargs", {}),
    )

    num_actions = env.action_space.n
    model = ActorCritic(in_channels=1, num_actions=num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["ppo"]["lr"]), eps=1e-5)

    # Transfer learning / Resume
    pretrained_path = args.pretrained or cfg["experiment"].get("pretrained_checkpoint")
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[+] Loading pretrained weights from {pretrained_path} for transfer learning...")
        ckpt = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # Rollout buffer
    ppo_cfg = cfg["ppo"]
    n_steps = ppo_cfg["n_steps"]
    buffer = RolloutBuffer(n_steps=n_steps, obs_shape=(1, 84, 84), device=device)

    total_timesteps = ppo_cfg["total_timesteps"]
    global_step = 0
    obs = env.reset()
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    best_eval_reward = -float("inf")
    start_time = time.time()

    print("[*] Training started...")
    while global_step < total_timesteps:
        buffer.reset()
        episode_rewards = []
        cur_ep_reward = 0.0

        # Collect trajectory
        for _ in range(n_steps):
            global_step += 1
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(state)

            next_obs, reward, done, info = env.step(action.item())
            cur_ep_reward += reward

            buffer.add(
                state=state.squeeze(0),
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )

            if done:
                episode_rewards.append(cur_ep_reward)
                cur_ep_reward = 0.0
                obs = env.reset()
                state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                state = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)

        # Estimate next value for GAE
        with torch.no_grad():
            _, _, _, next_val = model.get_action_and_value(state)

        buffer.compute_returns_and_advantages(
            compute_gae_fn=compute_gae,
            next_value=next_val.squeeze(),
            next_done=torch.tensor(done, device=device),
            gamma=float(ppo_cfg["gamma"]),
            gae_lambda=float(ppo_cfg["gae_lambda"]),
        )

        # PPO Update
        rollout_data = buffer.get_data()
        update_metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            rollout_data=rollout_data,
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            clip_ratio=float(ppo_cfg["clip_ratio"]),
            value_coef=float(ppo_cfg["value_coef"]),
            entropy_coef=float(ppo_cfg["entropy_coef"]),
            max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        )

        mean_ep_reward = np.mean(episode_rewards) if episode_rewards else cur_ep_reward
        sps = int(global_step / (time.time() - start_time + 1e-6))

        # 500k-step Health Check / Diagnostics (CN 22/11)
        if global_step % 500000 < n_steps:
            print(f"\n[DIAGNOSTICS @ {global_step} STEPS]")
            print(f" -> Mean Reward: {mean_ep_reward:.2f}")
            print(f" -> Approx KL: {update_metrics['approx_kl']:.6f} (Warning if > 0.03)")
            print(f" -> Clip Fraction: {update_metrics['clip_fraction']:.4f}")
            print(f" -> Entropy: {update_metrics['entropy']:.4f}")
            print(f" -> Speed (SPS): {sps} steps/sec\n")

        # Log metrics to progress.csv
        log_entry = {
            "step": global_step,
            "mean_reward": mean_ep_reward,
            "sps": sps,
            **update_metrics,
        }
        logger.log(log_entry)

        # Save latest checkpoint
        ckpt_path = os.path.join(checkpoint_dir, "latest.pt")
        torch.save({"model_state_dict": model.state_dict(), "global_step": global_step}, ckpt_path)

    env.close()
    print("[*] Training complete!")


if __name__ == "__main__":
    main()
