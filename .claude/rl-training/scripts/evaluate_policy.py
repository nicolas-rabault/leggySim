#!/usr/bin/env python3
"""Evaluate a trained policy headlessly — log metrics and record video.

Usage:
    uv run .claude/rl-training/scripts/evaluate_policy.py <wandb-run-path> [--output-dir <dir>] [--config <path>]
"""

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.wrappers import VideoRecorder


def parse_config(config_path):
    text = Path(config_path).read_text()

    task_section = text[text.index("## Task"):]
    task_match = re.search(r"^- Name:\s*(.+)$", task_section, re.MULTILINE)
    task_id = task_match.group(1).strip()

    actuators_match = re.search(r"^- Actuators:\s*\[(.+)\]$", text, re.MULTILINE)
    joints = [j.strip() for j in actuators_match.group(1).split(",")]

    scenarios = []
    in_scenarios = False
    for line in text.splitlines():
        if line.strip().startswith("- Scenarios:"):
            in_scenarios = True
            continue
        if in_scenarios:
            m = re.match(r"\s+- (\w+):\s*vx=([-+]?\d*\.?\d+),\s*vy=([-+]?\d*\.?\d+),\s*vz=([-+]?\d*\.?\d+),\s*steps=(\d+)", line)
            if m:
                scenarios.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5))))
            elif line.strip().startswith("- ") and not line.startswith("  "):
                break
    return task_id, joints, scenarios


def find_env_class(task_id):
    """Try to import the project's custom env class via task registration."""
    try:
        env_cfg = load_env_cfg(task_id, play=True)
        module_name = type(env_cfg).__module__
        pkg = module_name.split(".")[0]
        importlib.import_module(pkg + ".tasks")
    except Exception as e:
        print(f"[WARN] Could not import task module for {task_id}: {e}", file=sys.stderr)


def evaluate(run_path: str, output_dir: str, config_path: str):
    task_id, joints, scenarios = parse_config(config_path)
    find_env_class(task_id)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(task_id, play=True)
    rl_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.viewer.width = 1280
    env_cfg.viewer.height = 720
    env_cfg.viewer.distance = 1.0

    log_root = (Path("logs") / "rsl_rl" / rl_cfg.experiment_name).resolve()
    checkpoint_path, was_cached = get_wandb_checkpoint_path(log_root, Path(run_path))
    print(f"[INFO] Checkpoint: {checkpoint_path.name} ({'cached' if was_cached else 'downloaded'})")

    total_steps = sum(s for _, _, _, _, s in scenarios)

    env_cls_name = env_cfg.__class__.__name__.replace("Cfg", "")
    try:
        mod = importlib.import_module(type(env_cfg).__module__.rsplit(".", 1)[0])
        env_cls = getattr(mod, env_cls_name, None)
    except Exception as e:
        print(f"[WARN] Could not find env class {env_cls_name}: {e}", file=sys.stderr)
        env_cls = None

    if env_cls is None:
        from mjlab._train import ManagerBasedRlEnv
        env_cls = ManagerBasedRlEnv

    env = env_cls(cfg=env_cfg, device=device, render_mode="rgb_array")
    env.metadata["render_fps"] = 30
    env = VideoRecorder(
        env,
        video_folder=output_dir,
        step_trigger=lambda step: step == 0,
        video_length=total_steps,
        disable_logger=True,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

    from dataclasses import asdict
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(wrapped, asdict(rl_cfg), device=device)
    runner.load(str(checkpoint_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    obs = wrapped.get_observations()
    cmd_term = env.unwrapped.command_manager.get_term("twist")
    robot = env.unwrapped.scene["robot"]

    metrics = {}

    try:
        for cmd_name, vx, vy, vz, duration in scenarios:
            vel_errors_xy, vel_errors_yaw, torques, falls = [], [], [], 0
            actions_history, vel_x_history, vel_y_history, ang_z_history = [], [], [], []

            settle_steps = 100  # 1 second at 100Hz to reach steady state

            try:
                for step_i in range(duration):
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
                    actions_history.append(action[0].cpu().tolist())
                    vel_x_history.append(actual_vel[0].item())
                    vel_y_history.append(actual_vel[1].item())
                    ang_z_history.append(actual_ang)

                    if dones.any():
                        falls += 1
                        cmd_term.vel_command_b[:, 0] = vx
                        cmd_term.vel_command_b[:, 1] = vy
                        cmd_term.vel_command_b[:, 2] = vz
                        cmd_term.time_left[:] = 1e9
            except Exception as e:
                print(f"[WARN] Scenario '{cmd_name}' failed at step {len(vel_errors_xy)}: {e}", file=sys.stderr)
                if not vel_errors_xy:
                    metrics[cmd_name] = {"command": {"vx": vx, "vy": vy, "vz": vz}, "error": str(e)}
                    continue

            n = len(vel_errors_xy)
            avg_torques = [sum(t[i] for t in torques) / n for i in range(len(joints))]

            # Steady-state smoothness (after settling)
            ss = min(settle_steps, n // 2)
            ss_vel_x = vel_x_history[ss:]
            ss_vel_y = vel_y_history[ss:]
            ss_ang_z = ang_z_history[ss:]
            ss_actions = actions_history[ss:]

            import statistics
            vel_x_std = statistics.stdev(ss_vel_x) if len(ss_vel_x) > 1 else 0.0
            vel_y_std = statistics.stdev(ss_vel_y) if len(ss_vel_y) > 1 else 0.0
            ang_z_std = statistics.stdev(ss_ang_z) if len(ss_ang_z) > 1 else 0.0

            # Action jitter: mean of per-joint std of consecutive action differences
            action_jitter = 0.0
            if len(ss_actions) > 2:
                diffs = [[ss_actions[i+1][j] - ss_actions[i][j] for i in range(len(ss_actions)-1)] for j in range(len(ss_actions[0]))]
                action_jitter = sum(statistics.stdev(d) for d in diffs) / len(diffs)

            metrics[cmd_name] = {
                "command": {"vx": vx, "vy": vy, "vz": vz},
                "rms_vel_error_xy": (sum(e**2 for e in vel_errors_xy) / n)**0.5,
                "rms_vel_error_yaw": (sum(e**2 for e in vel_errors_yaw) / n)**0.5,
                "falls": falls,
                "mean_torque_per_joint": avg_torques,
                "steady_state_vel_x_std": vel_x_std,
                "steady_state_vel_y_std": vel_y_std,
                "steady_state_ang_z_std": ang_z_std,
                "steady_state_action_jitter": action_jitter,
            }
    finally:
        env.close()

    lines = [f"# Policy Evaluation — {run_path}", ""]
    for name, m in metrics.items():
        cmd = m["command"]
        lines.append(f"## {name} (vx={cmd['vx']}, vy={cmd['vy']}, vz={cmd['vz']})")
        if "error" in m:
            lines.append(f"- **FAILED**: {m['error']}")
            lines.append("")
            continue
        lines.append(f"- RMS velocity error XY: {m['rms_vel_error_xy']:.4f}")
        lines.append(f"- RMS velocity error yaw: {m['rms_vel_error_yaw']:.4f}")
        lines.append(f"- Falls: {m['falls']}")
        torque_str = ", ".join(f"{joints[i]}={m['mean_torque_per_joint'][i]:.3f}" for i in range(len(joints)))
        lines.append(f"- Mean torque: {torque_str}")
        lines.append(f"- Steady-state vel_x std: {m.get('steady_state_vel_x_std', 0):.4f}")
        lines.append(f"- Steady-state vel_y std: {m.get('steady_state_vel_y_std', 0):.4f}")
        lines.append(f"- Steady-state ang_z std: {m.get('steady_state_ang_z_std', 0):.4f}")
        lines.append(f"- Action jitter: {m.get('steady_state_action_jitter', 0):.4f}")
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
    parser.add_argument("run_path", help="wandb run path: entity/project/runs/run_id")
    parser.add_argument("--output-dir", required=True, help="Session run directory for output")
    parser.add_argument("--config", default=".claude/rl-training/config.md")
    args = parser.parse_args()
    evaluate(args.run_path, args.output_dir, args.config)


if __name__ == "__main__":
    main()
