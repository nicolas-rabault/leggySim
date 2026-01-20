import gymnasium as gym

gym.register(
    id="Mjlab-Stand-up-Flat-Leggy",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leggy_stand_up:LeggyStandUpEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.leggy_stand_up:LeggyStandUpRlCfg",
    },
)

gym.register(
    id="Mjlab-Stand-up-Flat-Leggy-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leggy_stand_up:LeggyStandUpEnvCfg_PLAY",
        "rl_cfg_entry_point": f"{__name__}.leggy_stand_up:LeggyStandUpRlCfg",
    },
)
