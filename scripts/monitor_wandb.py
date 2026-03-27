#!/usr/bin/env python3
"""Fetch wandb metrics and output a concise markdown summary.

Usage:
    uv run scripts/monitor_wandb.py <run-path> [--previous <file>]
    uv run scripts/monitor_wandb.py rabault-nicolas-leggy/mjlab/abc123
    uv run scripts/monitor_wandb.py rabault-nicolas-leggy/mjlab/abc123 --previous logs/training_session/run_001/monitor_001.md
"""

import argparse
import json
import sys
from pathlib import Path

import wandb


def get_run_status(run):
    return run.state  # "running", "finished", "killed", "crashed"


def fetch_latest_metrics(run, num_samples=5):
    rows = run.history(samples=num_samples, pandas=False)
    if not rows:
        return None, None
    latest = rows[-1]
    previous = rows[-2] if len(rows) >= 2 else None
    return latest, previous


def trend(current, previous, key):
    if previous is None or key not in previous or previous[key] is None:
        return ""
    if current.get(key) is None:
        return ""
    diff = current[key] - previous[key]
    if abs(diff) < 1e-6:
        return " (stable)"
    return f" ({'↑' if diff > 0 else '↓'} {abs(diff):.4f})"


def format_summary(run, latest, previous_check):
    status = get_run_status(run)
    step = latest.get("_step", "?")

    lines = [
        f"# WandB Monitor — {run.name}",
        f"**Status**: {status} | **Step**: {step}",
        "",
        "## Key Metrics",
    ]

    reward_keys = sorted(k for k in latest if k.startswith("Episode_Reward/") and latest[k] is not None)
    if reward_keys:
        lines.append("### Rewards")
        for k in reward_keys:
            name = k.replace("Episode_Reward/", "")
            val = latest[k]
            t = trend(latest, previous_check, k)
            lines.append(f"- **{name}**: {val:.4f}{t}")

    lines.append("")
    lines.append("### Training")
    for k in ["Train/mean_reward", "Train/mean_episode_length"]:
        if latest.get(k) is not None:
            t = trend(latest, previous_check, k)
            lines.append(f"- **{k.split('/')[-1]}**: {latest[k]:.2f}{t}")

    lines.append("")
    lines.append("### Terminations")
    term_keys = sorted(k for k in latest if k.startswith("Episode_Termination/") and latest[k] is not None)
    for k in term_keys:
        name = k.replace("Episode_Termination/", "")
        lines.append(f"- **{name}**: {latest[k]:.4f}")

    lines.append("")
    lines.append("### Curriculum")
    curr_keys = sorted(k for k in latest if k.startswith("Curriculum/") and latest[k] is not None)
    for k in curr_keys:
        name = k.replace("Curriculum/command_vel/", "")
        lines.append(f"- **{name}**: {latest[k]:.4f}")

    lines.append("")
    lines.append("### Torque")
    torque_keys = sorted(k for k in latest if k.startswith("Torque/") and latest[k] is not None)
    for k in torque_keys:
        name = k.replace("Torque/", "")
        lines.append(f"- **{name}**: {latest[k]:.4f}")

    lines.append("")
    lines.append("### Loss")
    for k in ["Loss/value_function", "Loss/surrogate", "Loss/entropy"]:
        if latest.get(k) is not None:
            t = trend(latest, previous_check, k)
            lines.append(f"- **{k.split('/')[-1]}**: {latest[k]:.6f}{t}")

    lines.append("")
    lines.append("### Velocity Tracking")
    for k in ["Metrics/twist/error_vel_xy", "Metrics/twist/error_vel_yaw"]:
        if latest.get(k) is not None:
            t = trend(latest, previous_check, k)
            lines.append(f"- **{k.split('/')[-1]}**: {latest[k]:.4f}{t}")

    return "\n".join(lines)


def load_previous_metrics(path):
    """Extract the raw metrics JSON from a previous monitor file if embedded."""
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text()
    marker = "<!-- RAW_METRICS:"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.index("-->", start)
        return json.loads(text[start:end])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_path", help="wandb run path: entity/project/run_id")
    parser.add_argument("--previous", help="Path to previous monitor output for trend comparison")
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.run_path)

    latest, wandb_previous = fetch_latest_metrics(run)
    if latest is None:
        print("No metrics available yet.")
        sys.exit(0)

    previous_check = None
    if args.previous:
        previous_check = load_previous_metrics(args.previous)
    if previous_check is None:
        previous_check = wandb_previous

    summary = format_summary(run, latest, previous_check)
    print(summary)

    # Embed raw metrics for next comparison
    print(f"\n<!-- RAW_METRICS:{json.dumps({k: v for k, v in latest.items() if v is not None})}-->")

    status = get_run_status(run)
    sys.exit(2 if status in ("killed", "crashed") else 0)


if __name__ == "__main__":
    main()
