# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from robot_lab.tasks.boying.mdp.commands import Go2RLGymCommand


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_term_name: str,
    num_steps_per_iter: int = 24,
) -> float:
    try:
        cmd_term = env.command_manager.get_term(command_term_name)
        cmd_cfg = cmd_term.cfg
    except LookupError:
        return 0.0

    current_iter = env.common_step_counter // num_steps_per_iter
    schedule = cmd_cfg.curriculum_schedule

    for i in range(len(schedule) - 1, -1, -1):
        stage = schedule[i]
        if current_iter >= stage['iter']:
            for t_name, active_range in cmd_cfg.ranges.items():
                hard_limit = cmd_cfg.terrain_max_ranges.get(t_name)
                if hard_limit is None:
                    continue

                def get_intersection(target_val, limit_val):
                    new_min = max(target_val[0], limit_val[0])
                    new_max = min(target_val[1], limit_val[1])
                    return (new_min, new_max)

                if 'lin_vel_x' in stage:
                    active_range.lin_vel_x = get_intersection(stage['lin_vel_x'], hard_limit.lin_vel_x)
                if 'lin_vel_y' in stage:
                    active_range.lin_vel_y = get_intersection(stage['lin_vel_y'], hard_limit.lin_vel_y)
                if 'ang_vel_yaw' in stage:
                    active_range.ang_vel_z = get_intersection(stage['ang_vel_yaw'], hard_limit.ang_vel_z)
                if 'heading' in stage and hasattr(active_range, 'heading'):
                    active_range.heading = get_intersection(stage['heading'], hard_limit.heading)
            schedule.pop(i)
            break

    last_key = list(cmd_cfg.ranges.keys())[-1]
    return cmd_cfg.ranges[last_key].lin_vel_x[1]


def gradual_ref_stand_modification(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial: float,
    final: float,
    start_it: int,
    end_it: int,
):
    current_it = env.common_step_counter // 24
    if current_it < start_it:
        return
    if current_it >= end_it:
        new = final
    else:
        new = (current_it - start_it) / (end_it - start_it) * (final - initial) + initial
    term = env.command_manager.get_term(term_name)
    term.cfg.rel_standing_envs = new


def gradual_reward_weight_modification(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    initial_weight: float,
    final_weight: float,
    start_it: int,
    end_it: int,
):
    """Curriculum that gradually modifies a reward weight between an initial and final value over a range of steps."""
    current_it = env.common_step_counter // 24
    if current_it < start_it:
        return
    if current_it >= end_it:
        new_weight = final_weight
    else:
        new_weight = (current_it - start_it) / (end_it - start_it) * (final_weight - initial_weight) + initial_weight
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)


def terrain_levels_vel_gym(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> float:
    terrain = env.scene.terrain
    command: Go2RLGymCommand = env.command_manager.get_term("base_velocity")

    max_move_dist = command.max_move_distance[env_ids]
    cmd_accum = command.commands_xy_accumulation[env_ids]

    resampling_time = command.cfg.resampling_time
    zero_prob = command.zero_command_prob

    terrain_cfg = terrain.cfg.terrain_generator
    sub_terrain_border_width = getattr(terrain_cfg, "sub_terrain_border_width", 0.0) or 0.0
    terrain_length = max(0.0, terrain_cfg.size[0] - 2.0 * sub_terrain_border_width)
    move_up = max_move_dist > terrain_length / 2
    target_dist = torch.norm(cmd_accum, dim=1) * (resampling_time * (1 - zero_prob))
    move_down = (max_move_dist < target_dist * 0.5) * ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())
