"""Export trained Leggy policy to ONNX and copy scene files for the web viewer."""

import argparse
import math
import shutil
from pathlib import Path

import onnx
import torch
import torch.nn as nn


DEFAULT_CHECKPOINT = "logs/rsl_rl/leggy/wandb_checkpoints/ckf6ortq/model_49999.pt"
DEFAULT_OUTPUT = "viewer/public/policy.onnx"

JOINT_NAMES = ["LhipY", "LhipX", "Lknee", "RhipY", "RhipX", "Rknee"]
OBS_NAMES = ["joint_pos", "joint_vel", "body_euler", "joint_torques", "base_lin_vel", "base_ang_vel", "command", "actions", "jump_command"]
DEFAULT_JOINT_POS = [0.22689280275926282, -0.6108652381980153, -0.6632251157578453, 0.22689280275926282, -0.6108652381980153, -0.6632251157578453]
ACTION_SCALE = 0.5
JOINT_STIFFNESS = [10, 10, 10, 10, 10, 10]
JOINT_DAMPING = [1, 1, 1, 1, 1, 1]
HISTORY_LENGTH = 5
OBS_SIZE = 37
DECIMATION = 2

LEGGY_DIR = Path("src/mjlab_leggy/leggy")
VIEWER_SCENES_DIR = Path("viewer/public/scenes")

# HOME_FRAME: stand pose from leggy_constants.py
_HIPY = 13 * math.pi / 180
_HIPX = -35 * math.pi / 180
_KNEE = (-38 + 35) * math.pi / 180  # motor_to_knee(-38 deg, -35 deg)
_PASSIVE_MOTOR = _KNEE + _HIPX       # knee_to_motor(knee, hipX)

# qpos layout: [root_pos(3), root_quat(4), LhipY, LhipX, Lknee, Lpassive2, LpassiveMotor, RhipY, RhipX, Rknee, Rpassive2, RpassiveMotor]
HOME_QPOS = [
    0, 0, 0.189,          # root position
    1, 0, 0, 0,           # root quaternion (identity)
    _HIPY, _HIPX, _KNEE,  # L: hipY, hipX, knee
    _KNEE, _PASSIVE_MOTOR, # L: passive2, passiveMotor
    _HIPY, _HIPX, _KNEE,  # R: hipY, hipX, knee
    _KNEE, _PASSIVE_MOTOR, # R: passive2, passiveMotor
]


class PolicyWithNormalizer(nn.Module):
    def __init__(self, mean, std, actor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.actor = actor

    def forward(self, obs):
        normalized = (obs - self.mean) / self.std
        return self.actor(normalized)


def build_actor(input_size=185):
    return nn.Sequential(
        nn.Linear(input_size, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 6),
    )


def copy_scene_files():
    VIEWER_SCENES_DIR.mkdir(parents=True, exist_ok=True)
    assets_dst = VIEWER_SCENES_DIR / "assets"
    assets_dst.mkdir(parents=True, exist_ok=True)

    # Copy robot.xml as-is (kp=10 matches training)
    shutil.copy2(LEGGY_DIR / "robot.xml", VIEWER_SCENES_DIR / "robot.xml")

    # Copy scene.xml with HOME_FRAME keyframe and training simulation options
    scene_xml = (LEGGY_DIR / "scene.xml").read_text()

    # Add simulation options matching training (timestep, integrator, solver)
    sim_options = '    <option timestep="0.005" iterations="10" ls_iterations="20" ccd_iterations="500" integrator="implicitfast" solver="Newton" cone="pyramidal"/>\n'
    scene_xml = scene_xml.replace('<include file="robot.xml" />', sim_options + '    <include file="robot.xml" />')

    # Add explicit friction to floor geom (MuJoCo default may differ across WASM/native)
    scene_xml = scene_xml.replace(
        '<geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" />',
        '<geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane" friction="1 0.005 0.0001"/>',
    )

    qpos_str = " ".join(f"{v:.10g}" for v in HOME_QPOS)
    keyframe = f'\n    <keyframe>\n        <key name="home" qpos="{qpos_str}"/>\n    </keyframe>'
    scene_xml = scene_xml.replace("</mujoco>", f"{keyframe}\n</mujoco>")
    (VIEWER_SCENES_DIR / "scene.xml").write_text(scene_xml)
    print(f"Copied robot.xml, scene.xml (with home keyframe) to {VIEWER_SCENES_DIR}")

    stl_count = 0
    for stl in (LEGGY_DIR / "assets").glob("*.stl"):
        shutil.copy2(stl, assets_dst / stl.name)
        stl_count += 1
    print(f"Copied {stl_count} STL files to {assets_dst}")


def main():
    parser = argparse.ArgumentParser(description="Export Leggy policy to ONNX + copy scene files")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_size = OBS_SIZE * HISTORY_LENGTH

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"]

    actor = build_actor(input_size)
    actor_state = {k.removeprefix("actor."): v for k, v in state.items() if k.startswith("actor.")}
    actor.load_state_dict(actor_state)

    mean = state["actor_obs_normalizer._mean"].squeeze(0)
    std = state["actor_obs_normalizer._std"].squeeze(0).clamp(min=1e-5)

    model = PolicyWithNormalizer(mean, std, actor)
    model.eval()

    dummy = torch.zeros(1, input_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(output_path),
        opset_version=17,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )

    onnx_model = onnx.load(str(output_path))
    metadata = {
        "joint_names": ",".join(JOINT_NAMES),
        "observation_names": ",".join(OBS_NAMES),
        "action_scale": str(ACTION_SCALE),
        "default_joint_pos": ",".join(str(v) for v in DEFAULT_JOINT_POS),
        "joint_stiffness": ",".join(str(v) for v in JOINT_STIFFNESS),
        "joint_damping": ",".join(str(v) for v in JOINT_DAMPING),
        "history_length": str(HISTORY_LENGTH),
        "obs_size": str(OBS_SIZE),
        "decimation": str(DECIMATION),
    }
    for key, value in metadata.items():
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = value

    onnx.save(onnx_model, str(output_path))
    print(f"Exported ONNX model to {output_path}")
    print(f"  Input: obs [{input_size}] (float32)")
    print(f"  Output: actions [{6}] (float32)")
    print(f"  Metadata: {len(metadata)} entries embedded")

    copy_scene_files()


if __name__ == "__main__":
    main()
