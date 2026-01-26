"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class ExportConfig:
    onnx_file: str = "output.onnx"
    agent: Literal["zero", "random", "trained"] = "trained"
    registry_name: str | None = None
    wandb_run_path: str | None = None
    checkpoint_file: str | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"

    # Internal flag used by demo script.
    _demo_mode: tyro.conf.Suppress[bool] = False


def run_export(task_id: str, cfg: ExportConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    # Check if this is a tracking task by checking for motion command.
    is_tracking_task = (
        env_cfg.commands is not None
        and "motion" in env_cfg.commands
        and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    )

    if is_tracking_task and cfg._demo_mode:
        # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.sampling_mode = "uniform"

    if is_tracking_task:
        assert env_cfg.commands is not None
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        if DUMMY_MODE:
            if not cfg.registry_name:
                raise ValueError(
                    "Tracking tasks require `registry_name` when using dummy agents."
                )
            # Check if the registry name includes alias, if not, append ":latest".
            registry_name = cfg.registry_name
            if ":" not in registry_name:
                registry_name = registry_name + ":latest"
            import wandb

            api = wandb.Api()
            artifact = api.artifact(registry_name)
            motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
        else:
            if cfg.motion_file is not None:
                print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
                motion_cmd.motion_file = cfg.motion_file
            else:
                import wandb

                api = wandb.Api()
                if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
                    raise ValueError(
                        "Tracking tasks require `motion_file` when using `checkpoint_file`, "
                        "or provide `wandb_run_path` so the motion artifact can be resolved."
                    )
                if cfg.wandb_run_path is not None:
                    wandb_run = api.run(str(cfg.wandb_run_path))
                    art = next(
                        (a for a in wandb_run.used_artifacts() if a.type == "motions"),
                        None,
                    )
                    if art is None:
                        raise RuntimeError("No motion artifact found in the run.")
                    motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError(
                    "`wandb_run_path` is required when `checkpoint_file` is not provided."
                )
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(cfg.wandb_run_path)
            )
            # Extract run_id and checkpoint name from path for display.
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "cached" if was_cached else "downloaded"
            print(
                f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
            )
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    if cfg.video and DUMMY_MODE:
        print(
            "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
        assert log_dir is not None  # log_dir is set in TRAINED_MODE block
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
        if cfg.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id) or OnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), map_location=device)
        policy = runner.get_inference_policy(device=device)

    # Monkey-patch get_base_metadata to support custom action types
    # We need to patch it in the exporter_utils module before it's imported elsewhere
    from mjlab.rl import exporter_utils

    def patched_get_base_metadata(env, run_path):
        """Modified version that works with custom action types like LeggyJointAction."""
        from mjlab.entity import Entity

        robot: Entity = env.scene["robot"]
        joint_action = env.action_manager.get_term("joint_pos")

        # Skip the isinstance check - trust that the action has the right interface
        # Build mapping from joint name to actuator ID
        joint_name_to_ctrl_id = {}
        for actuator in robot.spec.actuators:
            joint_name = actuator.target.split("/")[-1]
            joint_name_to_ctrl_id[joint_name] = actuator.id

        # Filter to only actuated joints (passive joints don't have actuators)
        actuated_joint_names = [jname for jname in robot.joint_names if jname in joint_name_to_ctrl_id]

        # Get actuator IDs in natural joint order (only for actuated joints)
        ctrl_ids_natural = [joint_name_to_ctrl_id[jname] for jname in actuated_joint_names]
        joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
        joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]

        return {
            "run_path": run_path,
            "joint_names": actuated_joint_names,  # Only actuated joints
            "joint_stiffness": joint_stiffness.tolist(),
            "joint_damping": joint_damping.tolist(),
            "default_joint_pos": robot.data.default_joint_pos[0, [robot.joint_names.index(jname) for jname in actuated_joint_names]].cpu().tolist(),
            "command_names": list(env.command_manager.active_terms),
            "observation_names": env.observation_manager.active_terms["policy"],
            "action_scale": joint_action._scale[0].cpu().tolist()
            if isinstance(joint_action._scale, torch.Tensor)
            else joint_action._scale,
        }

    # Patch the function in the module
    exporter_utils.get_base_metadata = patched_get_base_metadata

    # Now import the exporter - it will get the patched version
    from mjlab.tasks.velocity.rl import exporter

    # Also patch it directly in the exporter module since it may have already imported it
    exporter.get_base_metadata = patched_get_base_metadata

    from mjlab.tasks.velocity.rl.exporter import (
        export_velocity_policy_as_onnx,
        attach_onnx_metadata,
    )

    onnx_path = os.path.abspath(cfg.onnx_file)
    path = os.path.dirname(onnx_path)

    export_velocity_policy_as_onnx(
        runner.alg.policy,
        path=path,
        filename=onnx_path,
    )
    attach_onnx_metadata(
        runner.env.unwrapped,
        cfg.checkpoint_file,  # type: ignore
        path=path,
        filename=onnx_path,
    )
    print(f"Written {onnx_path}")

    env.close()


def main():
    # Parse first argument to choose the task.
    # Import tasks to populate the registry.
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
    agent_cfg = load_rl_cfg(chosen_task)

    args = tyro.cli(
        ExportConfig,
        args=remaining_args,
        default=ExportConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del remaining_args, agent_cfg

    run_export(chosen_task, args)


if __name__ == "__main__":
    main()
