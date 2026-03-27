#!/usr/bin/env python3
"""Evaluate a trained policy headlessly -- log metrics and record video.

Usage:
    uv run python scripts/evaluate_policy.py <wandb-run-path> [--output-dir <dir>]
    uv run python scripts/evaluate_policy.py rabault-nicolas-leggy/mjlab/xdvcvih3
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.wrappers import VideoRecorder
from mjlab_leggy.leggy.leggy_env import LeggyRlEnv

import mjlab_leggy.tasks  # noqa: F401 — registers Mjlab-Leggy

TASK_ID = "Mjlab-Leggy"

# (name, lin_vel_x, lin_vel_y, ang_vel_z, duration_steps)
TEST_COMMANDS = [
    ("stand", 0.0, 0.0, 0.0, 100),
    ("walk_forward", 0.5, 0.0, 0.0, 200),
    ("run_forward", 2.0, 0.0, 0.0, 200),
    ("turn_left", 0.5, 0.0, 1.0, 200),
    ("turn_right", 0.5, 0.0, -1.0, 200),
    ("side_step", 0.0, 0.5, 0.0, 200),
]


def evaluate(run_path: str, output_dir: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    rl_cfg = load_rl_cfg(TASK_ID)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = 1280
    env_cfg.viewer.height = 720
    env_cfg.viewer.distance = 1.0

    log_root = (Path("logs") / "rsl_rl" / rl_cfg.experiment_name).resolve()
    checkpoint_path, was_cached = get_wandb_checkpoint_path(log_root, Path(run_path))
    cached_str = "cached" if was_cached else "downloaded"
    print(f"[INFO] Checkpoint: {checkpoint_path.name} ({cached_str})")

    total_steps = sum(s for _, _, _, _, s in TEST_COMMANDS)
    env = LeggyRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env.metadata["render_fps"] = 30
    env = VideoRecorder(
        env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
    runner.load(str(checkpoint_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = wrapped.get_observations()
    cmd_term = env.unwrapped.command_manager.get_term("twist")
    robot = env.unwrapped.scene["robot"]

    metrics = {}

    for cmd_name, vx, vy, vz, duration in TEST_COMMANDS:
        vel_errors_xy = []
        vel_errors_yaw = []
        torques = []
        falls = 0

        for _ in range(duration):
            cmd_term.vel_command_b[:, 0] = vx
            cmd_term.vel_command_b[:, 1] = vy
            cmd_term.vel_command_b[:, 2] = vz
            cmd_term.time_left[:] = 1e9

            action = policy(obs)
            obs, _, dones, extras = wrapped.step(action)

            actual_vel = robot.data.root_link_lin_vel_b[0].cpu()
            actual_ang = robot.data.root_link_ang_vel_b[0, 2].cpu().item()
            torque = robot.data.actuator_force[0].cpu().abs().tolist()

            vel_errors_xy.append(((actual_vel[0].item() - vx)**2 + (actual_vel[1].item() - vy)**2)**0.5)
            vel_errors_yaw.append(abs(actual_ang - vz))
            torques.append(torque)

            if dones.any():
                falls += 1

        n = len(vel_errors_xy)
        avg_torques = [sum(t[i] for t in torques) / n for i in range(6)]
        metrics[cmd_name] = {
            "command": {"vx": vx, "vy": vy, "vz": vz},
            "rms_vel_error_xy": (sum(e**2 for e in vel_errors_xy) / n)**0.5,
            "rms_vel_error_yaw": (sum(e**2 for e in vel_errors_yaw) / n)**0.5,
            "falls": falls,
            "mean_torque": avg_torques,
        }

    env.close()

    joints = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]
    lines = [f"# Policy Evaluation -- {run_path}", ""]
    for name, m in metrics.items():
        cmd = m["command"]
        lines.append(f"## {name} (vx={cmd['vx']}, vy={cmd['vy']}, vz={cmd['vz']})")
        lines.append(f"- RMS velocity error XY: {m['rms_vel_error_xy']:.4f}")
        lines.append(f"- RMS velocity error yaw: {m['rms_vel_error_yaw']:.4f}")
        lines.append(f"- Falls: {m['falls']}")
        torque_str = ", ".join(f"{joints[i]}={m['mean_torque'][i]:.3f}" for i in range(6))
        lines.append(f"- Mean torque: {torque_str}")
        lines.append("")

    summary = "\n".join(lines)
    (output_dir / "eval_metrics.md").write_text(summary)
    (output_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(summary)

    video_files = list(output_dir.glob("*.mp4"))
    video_path = video_files[0] if video_files else None
    print(f"\nVideo: {video_path}")
    print(f"Metrics: {output_dir / 'eval_metrics.md'}")
    return str(video_path), str(output_dir / 'eval_metrics.md')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path: entity/project/run_id")
    parser.add_argument("--output-dir", default="logs/training_session/eval")
    args = parser.parse_args()
    evaluate(args.run_path, args.output_dir)


if __name__ == "__main__":
    main()
