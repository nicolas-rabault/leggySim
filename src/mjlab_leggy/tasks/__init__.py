from mjlab.tasks.registry import register_mjlab_task

from .leggy import leggy_env_cfg, leggy_rl_cfg

register_mjlab_task(
    task_id="Mjlab-Leggy",
    env_cfg=leggy_env_cfg(play=False),
    play_env_cfg=leggy_env_cfg(play=True),
    rl_cfg=leggy_rl_cfg(),
)
