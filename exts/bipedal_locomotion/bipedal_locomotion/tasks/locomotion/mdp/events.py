from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

def prepare_quantity_for_tron(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_radius = 0.127,
):
    asset: Articulation = env.scene[asset_cfg.name]
    env._foot_radius = foot_radius

def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.permanent_wrench_composer.set_forces_and_torques``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    asset.permanent_wrench_composer.reset()
    asset.instantaneous_wrench_composer.reset()

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(force_range[:, 0], force_range[:, 1], size, asset.device)
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(torque_range[:, 0], torque_range[:, 1], size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.permanent_wrench_composer.set_forces_and_torques(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)

def apply_external_force_torque_stochastic_additional(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.permanent_wrench_composer.set_forces_and_torques``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # clear the existing forces and torques
    # asset.permanent_wrench_composer.reset()
    asset.instantaneous_wrench_composer.reset()

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(force_range[:, 0], force_range[:, 1], size, asset.device)
    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(torque_range[:, 0], torque_range[:, 1], size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.permanent_wrench_composer.set_forces_and_torques(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)


def reset_robot_fallen_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    probability: float,
    base_height_range: tuple[float, float],
    pitch_range: tuple[float, float],
    roll_range: tuple[float, float],
    yaw_range: tuple[float, float],
    xy_range: tuple[float, float],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_noise_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset a subset of environments from side-lying or supine poses."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    if len(env_ids) == 0 or probability <= 0.0:
        return

    reset_mask = torch.rand(len(env_ids), device=asset.device) < probability
    fallen_env_ids = env_ids[reset_mask]
    if len(fallen_env_ids) == 0:
        return

    num_resets = len(fallen_env_ids)
    root_states = asset.data.default_root_state[fallen_env_ids].clone()

    positions = root_states[:, 0:3] + env.scene.env_origins[fallen_env_ids]
    positions[:, 0] += math_utils.sample_uniform(xy_range[0], xy_range[1], (num_resets,), asset.device)
    positions[:, 1] += math_utils.sample_uniform(xy_range[0], xy_range[1], (num_resets,), asset.device)
    positions[:, 2] = math_utils.sample_uniform(base_height_range[0], base_height_range[1], (num_resets,), asset.device)

    yaw = math_utils.sample_uniform(yaw_range[0], yaw_range[1], (num_resets,), asset.device)
    pose_selector = torch.rand(num_resets, device=asset.device)
    roll = torch.zeros(num_resets, device=asset.device)
    pitch = torch.zeros(num_resets, device=asset.device)

    left_ids = (pose_selector < 0.4).nonzero(as_tuple=False).squeeze(-1)
    right_ids = ((pose_selector >= 0.4) & (pose_selector < 0.8)).nonzero(as_tuple=False).squeeze(-1)
    supine_ids = (pose_selector >= 0.8).nonzero(as_tuple=False).squeeze(-1)

    if len(left_ids) > 0:
        roll[left_ids] = math_utils.sample_uniform(
            roll_range[0], roll_range[1], (int(left_ids.numel()),), asset.device
        )
    if len(right_ids) > 0:
        roll[right_ids] = math_utils.sample_uniform(
            -roll_range[1], -roll_range[0], (int(right_ids.numel()),), asset.device
        )
    if len(supine_ids) > 0:
        pitch[supine_ids] = math_utils.sample_uniform(
            pitch_range[0], pitch_range[1], (int(supine_ids.numel()),), asset.device
        )

    orientations_delta = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=asset.device)
    velocities = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (num_resets, 6), device=asset.device
    )

    joint_pos = asset.data.default_joint_pos[fallen_env_ids].clone()
    joint_vel = asset.data.default_joint_vel[fallen_env_ids].clone()
    joint_vel.zero_()

    cache_name = f"_{asset_cfg.name}_fallen_reset_joint_ids"
    leg_joint_ids = getattr(env, cache_name, None)
    if leg_joint_ids is None:
        leg_joint_ids = asset.find_joints(
            [
                "abad_L_Joint",
                "abad_R_Joint",
                "hip_L_Joint",
                "hip_R_Joint",
                "knee_L_Joint",
                "knee_R_Joint",
            ]
        )[0]
        leg_joint_ids = torch.as_tensor(leg_joint_ids, device=asset.device, dtype=torch.long)
        setattr(env, cache_name, leg_joint_ids)

    left_targets = torch.tensor([0.55, -0.10, 0.80, -0.35, 1.05, -0.80], device=asset.device)
    right_targets = torch.tensor([0.10, -0.55, 0.35, -0.80, 0.80, -1.05], device=asset.device)
    supine_targets = torch.tensor([0.20, -0.20, 0.90, -0.90, 1.10, -1.10], device=asset.device)

    if len(left_ids) > 0:
        joint_pos[left_ids.unsqueeze(-1), leg_joint_ids] = left_targets
    if len(right_ids) > 0:
        joint_pos[right_ids.unsqueeze(-1), leg_joint_ids] = right_targets
    if len(supine_ids) > 0:
        joint_pos[supine_ids.unsqueeze(-1), leg_joint_ids] = supine_targets

    joint_pos += math_utils.sample_uniform(
        joint_position_noise_range[0],
        joint_position_noise_range[1],
        joint_pos.shape,
        device=asset.device,
    )
    joint_pos = joint_pos.clamp_(
        asset.data.soft_joint_pos_limits[fallen_env_ids, :, 0],
        asset.data.soft_joint_pos_limits[fallen_env_ids, :, 1],
    )
    joint_vel = joint_vel.clamp_(
        -asset.data.soft_joint_vel_limits[fallen_env_ids],
        asset.data.soft_joint_vel_limits[fallen_env_ids],
    )

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=fallen_env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=fallen_env_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=fallen_env_ids)
    asset.set_joint_position_target(joint_pos, env_ids=fallen_env_ids)
    asset.set_joint_velocity_target(joint_vel, env_ids=fallen_env_ids)


def randomize_rigid_body_mass_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertias of the bodies (num_assets, num_bodies)
    inertias = asset.root_physx_view.get_inertias().clone()
    masses = asset.root_physx_view.get_masses().clone()

    masses = _randomize_prop_by_op(
        masses, mass_inertia_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    scale = masses / asset.root_physx_view.get_masses()
    inertias *= scale.unsqueeze(-1)

    asset.root_physx_view.set_masses(masses, env_ids)
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the center of mass (COM) of the bodies by adding, scaling, or setting random values for each dimension.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    coms = asset.root_physx_view.get_coms().clone()

    # Apply randomization to each dimension separately
    for dim in range(3):  # 0=x, 1=y, 2=z
        coms[..., dim] = _randomize_prop_by_op(
            coms[..., dim],
            com_distribution_params[dim],
            env_ids,
            body_ids,
            operation=operation,
            distribution=distribution,
        )

    asset.root_physx_view.set_coms(coms, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data
