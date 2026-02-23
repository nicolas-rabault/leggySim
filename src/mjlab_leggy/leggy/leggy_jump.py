"""Jump training for Leggy — command, rewards, curriculum, and configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from .leggy_constants import NUM_STEPS_PER_ENV

STANDING_HEIGHT = 0.175  # meters, from HOME_FRAME pos z
MIN_FLIGHT_STEPS = 3  # ~30ms at 100Hz, prevents micro-bounce exploit


# -- Command --


class JumpCommand(CommandTerm):
    cfg: JumpCommandCfg

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._jump_cmd = torch.zeros(self.num_envs, 1, device=self.device)
        self.achieved_flight = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.metrics["jump_active"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._jump_cmd

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
        self._jump_cmd[env_ids, 0] = (r < self.cfg.jump_probability).float()
        self.achieved_flight[env_ids] = False

    def _update_metrics(self) -> None:
        self.metrics["jump_active"] += self._jump_cmd[:, 0]

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class JumpCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    jump_probability: float = 0.0
    resampling_time_range: tuple[float, float] = (1.5, 3.0)

    def build(self, env) -> JumpCommand:
        return JumpCommand(self, env)


# -- Rewards --


class jump_height:
    """Per-step reward proportional to height during commanded flight."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.flight_steps = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: torch.Tensor):
        self.flight_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "feet_ground_contact",
        command_name: str = "jump",
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        contact = sensor.data.found.squeeze(-1) > 0
        both_off = ~contact[:, 0] & ~contact[:, 1]
        jump_cmd = env.command_manager.get_command(command_name)[:, 0] > 0.5

        in_flight = both_off & jump_cmd
        self.flight_steps = torch.where(
            in_flight,
            self.flight_steps + 1,
            torch.zeros_like(self.flight_steps),
        )

        # Mark flight achieved on command term (for jump_timeout)
        jump_term = env.command_manager.get_term(command_name)
        jump_term.achieved_flight |= self.flight_steps >= MIN_FLIGHT_STEPS

        base_z = env.scene["robot"].data.root_link_pos_w[:, 2]
        height_reward = (base_z / STANDING_HEIGHT - 1.0).clamp(min=0.0)

        return torch.where(
            self.flight_steps >= MIN_FLIGHT_STEPS,
            height_reward,
            torch.zeros_like(height_reward),
        )


class jump_landing:
    """Reward stable post-jump recovery for a fixed window after landing."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.in_jump_flight = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.landing_countdown = torch.zeros(
            env.num_envs, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: torch.Tensor):
        self.in_jump_flight[env_ids] = False
        self.landing_countdown[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str = "feet_ground_contact",
        command_name: str = "jump",
        post_landing_window: int = 10,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        contact = sensor.data.found.squeeze(-1) > 0
        any_on = contact[:, 0] | contact[:, 1]
        both_off = ~contact[:, 0] & ~contact[:, 1]
        jump_cmd = env.command_manager.get_command(command_name)[:, 0] > 0.5

        # Detect landing from jump flight
        just_landed = self.in_jump_flight & any_on
        self.landing_countdown = torch.where(
            just_landed & (self.landing_countdown == 0),
            torch.full_like(self.landing_countdown, post_landing_window),
            self.landing_countdown,
        )

        # Tick down
        active = self.landing_countdown > 0
        self.landing_countdown = (self.landing_countdown - active.long()).clamp(min=0)

        # Sticky flight: enter on both_off & jump_cmd, persist while in air
        self.in_jump_flight = both_off & (self.in_jump_flight | jump_cmd)

        return active.float()


def jump_timeout(
    env: ManagerBasedRlEnv,
    command_name: str = "jump",
    time_threshold: float = 0.3,
) -> torch.Tensor:
    """Penalty when jump command is about to expire without achieved flight."""
    jump_term = env.command_manager.get_term(command_name)
    jumping = jump_term.command[:, 0] > 0.5
    return (
        jumping & (jump_term.time_left < time_threshold) & ~jump_term.achieved_flight
    ).float()


def flight_penalty_jump_aware(
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "twist",
    run_threshold: float = 0.8,
    jump_command_name: str = "jump",
) -> torch.Tensor:
    """Flight penalty that allows flight during jump commands."""
    jumping = env.command_manager.get_command(jump_command_name)[:, 0] > 0.5

    sensor: ContactSensor = env.scene[sensor_name]
    contact = sensor.data.found.squeeze(-1) > 0
    both_in_air = (~contact[:, 0] & ~contact[:, 1]).float()

    vel_cmd = env.command_manager.get_command(command_name)[:, :2]
    scale = torch.clamp(1.0 - torch.norm(vel_cmd, dim=1) / run_threshold, min=0.0)
    penalty = both_in_air * scale

    return torch.where(jumping, torch.zeros_like(penalty), penalty)


# -- Observation --


def jump_command_obs(
    env, command_name: str = "jump", asset_cfg=None
) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


# -- Curriculum --

_N_JUMP_STAGES = 4
_FIRST_JUMP_ITERATION = 70_000
_LAST_JUMP_ITERATION = 90_000
_FIRST_JUMP_STEP = _FIRST_JUMP_ITERATION * NUM_STEPS_PER_ENV
_LAST_JUMP_STEP = _LAST_JUMP_ITERATION * NUM_STEPS_PER_ENV
_JUMP_INITIAL = {"jump_probability": 0.15, "height_w": 10.0, "landing_w": 1.5, "timeout_w": -0.5}
_JUMP_FINAL = {"jump_probability": 0.25, "height_w": 15.0, "landing_w": 2.0, "timeout_w": -1.0}

JUMP_STAGES = []
for _i in range(_N_JUMP_STAGES):
    _t = _i / (_N_JUMP_STAGES - 1)
    _stage = {"step": round(_FIRST_JUMP_STEP + _t * (_LAST_JUMP_STEP - _FIRST_JUMP_STEP))}
    for _key in _JUMP_INITIAL:
        _stage[_key] = round(_JUMP_INITIAL[_key] + _t * (_JUMP_FINAL[_key] - _JUMP_INITIAL[_key]), 4)
    JUMP_STAGES.append(_stage)


def jump_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    jump_stages: list[dict],
) -> dict[str, float]:
    del env_ids
    jump_term = env.command_manager.get_term("jump")

    for stage in jump_stages:
        if env.common_step_counter > stage["step"]:
            jump_term.cfg.jump_probability = stage["jump_probability"]
            env.reward_manager.get_term_cfg("jump_height").weight = stage["height_w"]
            env.reward_manager.get_term_cfg("jump_landing").weight = stage["landing_w"]
            env.reward_manager.get_term_cfg("jump_timeout").weight = stage["timeout_w"]

    return {
        "jump_probability": jump_term.cfg.jump_probability,
        "height_weight": env.reward_manager.get_term_cfg("jump_height").weight,
    }


# -- Configure --


def configure_jump(cfg):
    """Add jump training to environment config."""
    # 1. Jump command (starts disabled)
    cfg.commands["jump"] = JumpCommandCfg()

    # 2. Jump observation for policy and critic
    for group in ("policy", "critic"):
        cfg.observations[group].terms["jump_command"] = ObservationTermCfg(
            func=jump_command_obs,
            params={"command_name": "jump"},
        )

    # 3. Jump rewards (all start at weight=0, enabled by curriculum)
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=jump_height,
        weight=0.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "jump"},
    )
    cfg.rewards["jump_landing"] = RewardTermCfg(
        func=jump_landing,
        weight=0.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "jump"},
    )
    cfg.rewards["jump_timeout"] = RewardTermCfg(
        func=jump_timeout,
        weight=0.0,
        params={"command_name": "jump"},
    )

    # 4. Replace flight_penalty with jump-aware version
    cfg.rewards["flight_penalty"] = RewardTermCfg(
        func=flight_penalty_jump_aware,
        weight=-2.0,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "run_threshold": 0.8,
            "jump_command_name": "jump",
        },
    )

    # 5. Jump curriculum
    cfg.curriculum["jump"] = CurriculumTermCfg(
        func=jump_curriculum,
        params={"jump_stages": JUMP_STAGES},
    )
