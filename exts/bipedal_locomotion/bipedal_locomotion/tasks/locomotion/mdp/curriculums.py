from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def modify_event_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    value: Any | SceneEntityCfg,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies a parameter of an event at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the event term.
        param_name: The name of the event term parameter.
        value: The new value for the event term parameter.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.event_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.params[param_name] = value
        env.event_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def disable_termination(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
) -> torch.Tensor:
    """Curriculum that modifies the push velocity range at a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the termination term.
        num_steps: The number of steps after which the change should be applied.

    Returns:
        torch.Tensor: Whether the parameter has already been modified or not.
    """
    env.command_manager.num_envs
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.termination_manager.get_term_cfg(term_name)
        # Remove term settings
        term_cfg.params = dict()
        term_cfg.func = lambda env: torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        env.termination_manager.set_term_cfg(term_name, term_cfg)
        return torch.ones(1)
    return torch.zeros(1)


def jump_probability_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    start_prob: float,
    end_prob: float,
    start_iteration: int,
    end_iteration: int,
    num_steps_per_env: int = 24,
) -> torch.Tensor:
    """Linearly ramp jump_probability from start_prob to end_prob over [start_iteration, end_iteration].

    Uses env.common_step_counter / num_steps_per_env to approximate the current
    RL iteration, since common_step_counter increments every env step.

    Returns:
        torch.Tensor: Current jump_probability as a scalar tensor (for logging).
    """
    iteration = env.common_step_counter / num_steps_per_env
    if iteration <= start_iteration:
        prob = start_prob
    elif iteration >= end_iteration:
        prob = end_prob
    else:
        alpha = (iteration - start_iteration) / (end_iteration - start_iteration)
        prob = start_prob + alpha * (end_prob - start_prob)
    term = env.command_manager.get_term(command_name)
    term.cfg.jump_probability = prob
    return torch.tensor([prob], dtype=torch.float32)


def jump_assist_force_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    force_max: float,
    decay_start_iteration: int,
    decay_per_1000_iter: float,
    num_steps_per_env: int = 24,
) -> torch.Tensor:
    """Exponentially decay the upward assist force applied during jump.

    Before decay_start_iteration: assist_force_max = force_max.
    After that: multiplied by (1 - decay_per_1000_iter)^(n/1000) each iteration,
    where n = iterations elapsed since decay_start_iteration.
    Once the force reaches 0 (scale < 1e-3), it is clamped to 0.

    Returns:
        torch.Tensor: Current assist force magnitude (for logging).
    """
    iteration = env.common_step_counter / num_steps_per_env
    if iteration < decay_start_iteration:
        scale = 1.0
    else:
        n_thousands = (iteration - decay_start_iteration) / 1000.0
        scale = (1.0 - decay_per_1000_iter * n_thousands)
        if scale < 1e-3:
            scale = 0.0

    current_force = force_max * scale
    term = env.command_manager.get_term(command_name)
    term.cfg.assist_force_max = current_force
    return torch.tensor([current_force], dtype=torch.float32)


def fallen_reset_probability_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_prob: float,
    end_prob: float,
    start_iteration: int,
    end_iteration: int,
    num_steps_per_env: int = 24,
) -> torch.Tensor:
    """Linearly ramp the reset probability for the fallen-state reset event."""
    del env_ids
    iteration = env.common_step_counter / num_steps_per_env
    if iteration <= start_iteration:
        prob = start_prob
    elif iteration >= end_iteration:
        prob = end_prob
    else:
        alpha = (iteration - start_iteration) / (end_iteration - start_iteration)
        prob = start_prob + alpha * (end_prob - start_prob)

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["probability"] = prob
    env.event_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor([prob], dtype=torch.float32)


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "rew_lin_vel_xy",
) -> torch.Tensor:
    """Expand lin_vel_x/y command ranges when tracking reward is high enough.

    Every max_episode_length steps, if mean episode reward exceeds 80% of the
    reward weight, expand ranges by ±0.1 (clamped to limit_ranges).

    Returns:
        torch.Tensor: Max absolute velocity endpoint (for logging).
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    max_x = max(ranges.lin_vel_x, key=abs)
    max_y = max(ranges.lin_vel_y, key=abs)
    max_endpoint = max(max_x, max_y, key=abs)
    return torch.tensor(abs(max_endpoint), device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "rew_ang_vel_z",
) -> torch.Tensor:
    """Expand ang_vel_z command range when tracking reward is high enough.

    Only evaluates reward from low-speed envs (lin_speed < 1.5) to avoid
    conflating turning quality with straight-line speed.

    Returns:
        torch.Tensor: Max absolute angular velocity endpoint (for logging).
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    lin_vel_xy = env.scene["robot"].data.root_lin_vel_b[env_ids, :2]
    lin_speed = torch.norm(lin_vel_xy, dim=1)
    low_speed_mask = lin_speed < 1.5
    if low_speed_mask.sum() > 0:
        env_ids_tensor = torch.as_tensor(env_ids, device=env.device)
        low_speed_ids = env_ids_tensor[low_speed_mask]
        reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][low_speed_ids]) / env.max_episode_length_s
    else:
        reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.7:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    max_endpoint = abs(max(ranges.ang_vel_z, key=abs))
    return torch.tensor(max_endpoint, device=env.device)


def reward_weight_abs_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    reward_term_name: str,
    reward_threshold_ratio: float,
    step: float,
    max_abs_weight: float,
    num_steps_per_env: int = 24,
    min_interval_iterations: int = 0,
) -> torch.Tensor:
    """Increase the absolute weight of a reward term with an optional cooldown.

    The adjustment is evaluated on episode boundaries once the tracked reward
    exceeds the configured threshold and training has passed the initial warmup.
    After each successful weight change, the term can be updated again only
    after ``min_interval_iterations`` PPO iterations.
    """
    target_term = env.reward_manager.get_term_cfg(reward_term_name)
    adjusted_term = env.reward_manager.get_term_cfg(term_name)

    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    current_iteration = int(env.common_step_counter / num_steps_per_env)
    if target_term.weight >= 0.0:
        reward_threshold = abs(target_term.weight) * reward_threshold_ratio
    else:
        reward_threshold = -abs(target_term.weight) * reward_threshold_ratio

    last_update_attr = "_reward_weight_abs_curriculum_last_update_iter"
    last_update_iters = getattr(env, last_update_attr, {})
    state_key = term_name
    last_update_iteration = last_update_iters.get(state_key)
    interval_satisfied = (
        last_update_iteration is None
        or current_iteration - last_update_iteration >= min_interval_iterations
    )

    if (
        env.common_step_counter % env.max_episode_length == 0
        and reward > reward_threshold
        and abs(reward) > 1e-6
        and current_iteration > 1000
        and interval_satisfied
    ):
        sign = -1.0 if adjusted_term.weight < 0.0 else 1.0
        new_abs_weight = min(abs(adjusted_term.weight) + step, max_abs_weight)
        if new_abs_weight != abs(adjusted_term.weight):
            adjusted_term.weight = sign * new_abs_weight
            env.reward_manager.set_term_cfg(term_name, adjusted_term)
            last_update_iters[state_key] = current_iteration
            setattr(env, last_update_attr, last_update_iters)

    return torch.tensor(abs(adjusted_term.weight), device=env.device)
