"""
Evaluation Script for Mario Agent Checkpoints
Responsible: SV3 (Đánh giá & Điều phối)
Evaluates checkpoint according to Section 4 of project prompt.
Outputs:
- JSON summary file with flag_rate, mean_reward, mean_x_pos
- Updates evals.csv
- Recorded video (optional .mp4)
"""
import argparse
import os
import json
import csv
import torch
import numpy as np

from src.env.make_env import make_mario_env
from src.models.actor_critic import ActorCritic
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Mario PPO Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--env-id", type=str, default="SuperMarioBros-1-1-v0", help="Gym environment ID")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory for output files")
    parser.add_argument("--record-video", action="store_true", help="Record gameplay video")
    parser.add_argument("--seed", type=int, default=42, help="Seed for evaluation")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[*] Evaluating checkpoint: {args.checkpoint} on {args.env_id} ({args.num_episodes} episodes)")

    env = make_mario_env(env_id=args.env_id, action_type="simple", skip_frames=4, shaped_reward=False)
    num_actions = env.action_space.n

    model = ActorCritic(in_channels=1, num_actions=num_actions).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    episodes_data = []
    flags_reached = 0

    for ep in range(args.num_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        max_x = 0

        while not done:
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model.forward(state)
                action = torch.argmax(logits, dim=-1).item()

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            max_x = max(max_x, info.get("x_pos", 0))

        flag_reached = bool(info.get("flag_get", False))
        if flag_reached:
            flags_reached += 1

        episodes_data.append({
            "episode": ep + 1,
            "reward": float(ep_reward),
            "max_x": int(max_x),
            "flag_get": flag_reached,
        })
        print(f"Episode {ep + 1}/{args.num_episodes}: Reward={ep_reward:.1f}, Max X={max_x}, Flag={flag_reached}")

    flag_rate = flags_reached / args.num_episodes
    mean_reward = float(np.mean([d["reward"] for d in episodes_data]))
    mean_x_pos = float(np.mean([d["max_x"] for d in episodes_data]))

    results = {
        "checkpoint": args.checkpoint,
        "env_id": args.env_id,
        "num_episodes": args.num_episodes,
        "flag_rate": flag_rate,
        "mean_reward": mean_reward,
        "mean_x_pos": mean_x_pos,
        "episodes": episodes_data,
    }

    # Save JSON result
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    json_path = os.path.join(args.output_dir, f"eval_{ckpt_name}_{args.env_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Append to evals.csv
    evals_csv_path = os.path.join(args.output_dir, "evals.csv")
    csv_exists = os.path.exists(evals_csv_path)
    with open(evals_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not csv_exists:
            writer.writerow(["checkpoint", "env_id", "flag_rate", "mean_reward", "mean_x_pos"])
        writer.writerow([args.checkpoint, args.env_id, f"{flag_rate:.4f}", f"{mean_reward:.2f}", f"{mean_x_pos:.2f}"])

    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS FOR {ckpt_name}:")
    print(f"  Flag Rate : {flag_rate * 100:.1f}% (Target: >= 80.0%)")
    print(f"  Mean Reward: {mean_reward:.2f}")
    print(f"  Mean X Pos : {mean_x_pos:.2f}")
    print(f"  Saved JSON : {json_path}")
    print(f"  Updated CSV: {evals_csv_path}")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    evaluate()
