# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Package containing task implementations for boying robot environments."""

import gymnasium as gym

from isaaclab_tasks.utils import import_packages

gym.register(
    id="RobotLab-Boying-v0",
    entry_point="robot_lab.tasks.boying.env.boying_env:BoyingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BoyingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:MoECTSRunnerCfg",
    },
)

_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
