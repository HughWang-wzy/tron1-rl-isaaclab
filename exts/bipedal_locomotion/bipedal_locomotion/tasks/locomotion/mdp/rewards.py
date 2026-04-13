"""This sub-module contains the reward functions that can be used for LimX Point Foot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import distributions
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def stay_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for staying alive."""
    return torch.ones(env.num_envs, device=env.device)


def track_lin_vel_xy_exp_adaptive(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_scale: float = 0.35,
) -> torch.Tensor:
    """Reward xy velocity tracking with a command-scaled tolerance.

    Low-speed commands keep the original strict ``std``. For large commanded
    speeds, the tolerance grows with command magnitude so the reward does not
    collapse to near zero before the policy reaches high-speed tracking.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    lin_vel_error = torch.sum(torch.square(command - asset.data.root_lin_vel_b[:, :2]), dim=1)

    command_speed = torch.norm(command, dim=1)
    adaptive_std = torch.maximum(torch.full_like(command_speed, std), command_scale * command_speed)

    return torch.exp(-lin_vel_error / torch.square(adaptive_std.clamp_min(1e-6)))

def foot_landing_vel(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        foot_radius: float,
        about_landing_threshold: float,
) -> torch.Tensor:
    """Penalize high foot landing velocities"""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    z_vels = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 0.1

    foot_heights = torch.clip(
    asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain

    about_to_land = (foot_heights < about_landing_threshold) & (~contacts) & (z_vels < 0.0)
    landing_z_vels = torch.where(about_to_land, z_vels, torch.zeros_like(z_vels))
    reward = torch.sum(torch.square(landing_z_vels), dim=1)
    return reward

def joint_powers_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel"""

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


def no_fly(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]

    contacts = latest_contact_forces > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1

    return 1.0 * single_contact


def unbalance_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize if the feet air time variance exceeds the balance threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    return torch.var(contact_sensor.data.last_air_time[:, sensor_cfg.body_ids], dim=-1)


def unbalance_feet_height(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the variance of feet maximum height using sensor positions."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    feet_positions = contact_sensor.data.pos_w[:, sensor_cfg.body_ids]

    if feet_positions is None:
        return torch.zeros(env.num_envs)

    feet_heights = feet_positions[:, :, 2]
    max_feet_heights = torch.max(feet_heights, dim=-1)[0]
    height_variance = torch.var(max_feet_heights, dim=-1)
    return height_variance


# def feet_distance(
#     env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
# ) -> torch.Tensor:
#     """Penalize if the distance between feet is below a minimum threshold."""

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions = asset.data.joint_pos[sensor_cfg.body_ids]

#     if feet_positions is None:
#         return torch.zeros(env.num_envs)

#     # feet distance on x-y plane
#     feet_distance = torch.norm(feet_positions[0, :2] - feet_positions[1, :2], dim=-1)

#     return torch.clamp(0.1 - feet_distance, min=0.0)


def feet_distance(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                  feet_links_name: list[str]=["foot_[RL]_Link"],
                  min_feet_distance: float = 0.1,
                  max_feet_distance: float = 1.0,)-> torch.Tensor:
    # Penalize base height away from target
    asset: Articulation = env.scene[asset_cfg.name]
    feet_links_idx = asset.find_bodies(feet_links_name)[0]
    feet_pos = asset.data.body_link_pos_w[:,feet_links_idx]
    # feet distance on x-y plane
    feet_distance = torch.norm(feet_pos[:, 0, :2] - feet_pos[:, 1, :2], dim=-1)
    reward = torch.clip(min_feet_distance - feet_distance, 0, 1)
    reward += torch.clip(feet_distance - max_feet_distance, 0, 1)
    return reward

def nominal_foot_position(env: ManagerBasedRLEnv, command_name: str,
                          base_height_target: float,
                           asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Compute the nominal foot position"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_center_b = torch.mean(feet_pos_b[:, :, :3], dim=1)
    base_height_error = torch.abs((feet_center_b[:, 2] - env._foot_radius + base_height_target))

    reward = torch.exp(-base_height_error / std**2)
    return reward

def leg_symmetry(env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    leg_symmetry_err = torch.abs(feet_pos_b[:, 0, 1]) - torch.abs(feet_pos_b[:, 1, 1])

    return torch.exp(-leg_symmetry_err ** 2 / std**2)

def same_feet_x_position(env: ManagerBasedRLEnv,
                  asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward regulate abad joint position."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    # assert (compute_rotation_distance(asset.data.root_com_quat_w, asset.data.root_link_quat_w) < 0.1).all()
    base_pos = asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, 2, -1)
    feet_pos_b = math_utils.quat_apply_inverse(
        base_quat,
        feet_pos_w - base_pos,
    )
    feet_x_distance = torch.abs(feet_pos_b[:, 0, 0] - feet_pos_b[:, 1, 0])
    # return torch.exp(-feet_x_distance / 0.2)
    return feet_x_distance

def keep_ankle_pitch_zero_in_air(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=["ankle_[LR]_Link"]),
    force_threshold: float = 2.0,
    pitch_scale: float = 0.2
) -> torch.Tensor:
    """Reward for keeping ankle pitch angle close to zero when foot is in the air.
    
    Args:
        env: The environment object.
        asset_cfg: Configuration for the robot asset containing DOF positions.
        sensor_cfg: Configuration for the contact force sensor.
        force_threshold: Threshold value for contact detection (in Newtons).
        pitch_scale: Scaling factor for the exponential reward.
        
    Returns:
        The computed reward tensor.
    """
    asset = env.scene[asset_cfg.name]
    contact_forces_history = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    current_contact = torch.norm(contact_forces_history[:, -1], dim=-1) > force_threshold
    last_contact = torch.norm(contact_forces_history[:, -2], dim=-1) > force_threshold
    contact_filt = torch.logical_or(current_contact, last_contact)
    ankle_pitch_left = torch.abs(asset.data.joint_pos[:, 3]) * ~contact_filt[:, 0]
    ankle_pitch_right = torch.abs(asset.data.joint_pos[:, 7]) * ~contact_filt[:, 1]
    weighted_ankle_pitch = ankle_pitch_left + ankle_pitch_right
    return torch.exp(-weighted_ankle_pitch / pitch_scale)

def no_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize if both feet are not in contact with the ground.
    """

    # Access the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the latest contact forces in the z direction (upward direction)
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]  # shape: (env_num, 2)

    # Determine if each foot is in contact
    contacts = latest_contact_forces > 1.0  # Returns a boolean tensor where True indicates contact

    return (torch.sum(contacts.float(), dim=1) == 0).float()


def stand_still(
    env,
    lin_threshold: float = 0.05,
    ang_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    target_joint_positions: dict[str, float] | None = None,
    joint_pos_weight: float = 1.0,
    sensor_cfg: SceneEntityCfg | None = None,
    no_contact_threshold: float = 1.0,
    no_contact_weight: float = 1.0,
) -> torch.Tensor:
    """
    Penalize base motion when command velocities are near zero.

    Optionally also penalize selected joints deviating from target positions
    while the robot is commanded to stand still. If a contact sensor is
    provided, also penalize the case where both feet are off the ground.
    """

    asset = env.scene[asset_cfg.name]
    base_lin_vel = asset.data.root_lin_vel_w[:, :2]
    base_ang_vel = asset.data.root_ang_vel_w[:, -1]

    commands = env.command_manager.get_command(command_name)

    lin_commands = commands[:, :2]
    ang_commands = commands[:, 2]
    lin_cmd_is_still = torch.norm(lin_commands, dim=1, keepdim=True) < lin_threshold
    ang_cmd_is_still = torch.abs(ang_commands) < ang_threshold
    stand_still_mask = (lin_cmd_is_still.squeeze(-1) & ang_cmd_is_still).float()

    reward_lin = torch.sum(torch.abs(base_lin_vel) * lin_cmd_is_still, dim=-1)
    reward_ang = torch.abs(base_ang_vel) * ang_cmd_is_still
    total_reward = reward_lin + reward_ang

    if target_joint_positions:
        joint_error = torch.zeros(env.num_envs, device=env.device)
        for joint_name, target_pos in target_joint_positions.items():
            joint_ids, _ = asset.find_joints(joint_name)
            joint_error += torch.abs(asset.data.joint_pos[:, joint_ids[0]] - target_pos)

        total_reward += joint_pos_weight * joint_error * stand_still_mask

    if sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, 2]
        contacts = latest_contact_forces > no_contact_threshold
        both_feet_off_ground = (torch.sum(contacts.float(), dim=1) == 0).float()
        total_reward += no_contact_weight * both_feet_off_ground * stand_still_mask

    return total_reward


# def feet_regulation(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     desired_body_height: float = 0.65,
# ) -> torch.Tensor:
#     """Penalize if the feet are not in contact with the ground.

#     Args:
#         env: The environment object.
#         sensor_cfg: The configuration of the contact sensor.
#         desired_body_height: The desired body height used for normalization.

#     Returns:
#         A tensor representing the feet regulation penalty for each environment.
#     """

#     asset: Articulation = env.scene[asset_cfg.name]

#     feet_positions_z = asset.data.joint_pos[sensor_cfg.body_ids, 2]

#     feet_vel_xy = asset.data.joint_vel[sensor_cfg.body_ids, :2]

#     vel_norms_xy = torch.norm(feet_vel_xy, dim=-1)

#     exp_term = torch.exp(-feet_positions_z / (0.025 * desired_body_height))

#     r_fr = torch.sum(vel_norms_xy**2 * exp_term, dim=-1)

#     return r_fr

def feet_regulation(env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    foot_radius: float,
    base_height_target: float,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_height = torch.clip(
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - foot_radius, 0, 1
    )  # TODO: change to the height relative to the vertical projection of the terrain
    feet_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    height_scale = torch.exp(-feet_height / base_height_target)
    reward = torch.sum(height_scale * torch.square(torch.norm(feet_vel_xy, dim=-1)), dim=1)
    return reward


def base_height_rough_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        Currently, it assumes a flat terrain, i.e. the target height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = asset.data.root_pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[:, :, 2]
    # sensor.data.ray_hits_w can be inf, so we clip it to avoid NaN
    height = torch.nan_to_num(height, nan=target_height, posinf=target_height, neginf=target_height)
    return torch.square(height.mean(dim=1) - target_height)


def base_com_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.abs(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def track_base_height_from_command(
    env: ManagerBasedRLEnv,
    command_name: str = "height_command",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Track commanded base height using exp kernel.

    Reads target_height from the height command.
    Returns exp(-error^2 / sigma^2), rewarding being close to the target.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    height_cmd = env.command_manager.get_command(command_name)
    target_height = height_cmd[:, 0]  # shape: (num_envs,)
    error = asset.data.root_pos_w[:, 2] - target_height
    return torch.exp(-torch.square(error) / (sigma ** 2))


class GaitReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # extract the used quantities (to enable type-hinting)
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.swing_height_scale = float(cfg.params.get("swing_height_scale", 0.0))
        self.foot_radius = float(cfg.params.get("foot_radius", 0.05))
        self.max_swing_height = float(cfg.params.get("max_swing_height", 0.15))
        self.velocity_command_name = cfg.params.get("velocity_command_name", "base_velocity")
        self.standstill_threshold = float(cfg.params.get("standstill_threshold", 0.1))
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
        swing_height_scale,
        foot_radius,
        max_swing_height,
    ) -> torch.Tensor:
        """Compute the reward.

        The reward combines force-based and velocity-based terms to encourage desired gait patterns.

        Args:
            env: The RL environment instance.

        Returns:
            The reward value.
        """

        gait_params = env.command_manager.get_command(self.command_name)

        # Update contact targets
        desired_contact_states = self.compute_contact_targets(gait_params)

        # Standstill override: when velocity command is near zero, all feet should be in contact
        vel_cmd = env.command_manager.get_command(self.velocity_command_name)
        standstill_mask = torch.norm(vel_cmd[:, :2], dim=1) < self.standstill_threshold
        desired_contact_states[standstill_mask] = 1.0

        # Force-based reward
        foot_forces = torch.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1)
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward
        foot_velocities = torch.norm(self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1)
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        # Swing foot height reward: reward feet for lifting higher during swing phase
        height_reward = torch.zeros_like(force_reward)
        if self.swing_height_scale > 0.0:
            foot_heights = torch.clamp(
                self.asset.data.body_pos_w[:, self.asset_cfg.body_ids, 2] - self.foot_radius, min=0.0
            )
            clamped_heights = torch.clamp(foot_heights, max=self.max_swing_height)
            # (1 - desired_contact) = swing phase weight
            height_reward = self.swing_height_scale * torch.sum(
                (1 - desired_contact_states) * clamped_heights, dim=1
            
            )

        # Combine rewards
        total_reward = force_reward + velocity_reward + height_reward
        return total_reward

    def compute_contact_targets(self, gait_params):
        """Calculate desired contact states for the current timestep."""
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        assert torch.all(frequencies > 0), "Frequencies must be positive"
        assert torch.all((offsets >= 0) & (offsets <= 1)), "Offsets must be between 0 and 1"
        assert torch.all((durations > 0) & (durations < 1)), "Durations must be between 0 and 1"

        gait_indices = torch.remainder(self._env.episode_length_buf * self.dt * frequencies, 1.0)

        # Calculate foot indices
        foot_indices = torch.remainder(
            torch.cat(
                [gait_indices.view(self.num_envs, 1), (gait_indices + offsets + 1).view(self.num_envs, 1)],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Adjust foot indices based on phase
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        foot_indices[swing_idxs] = 0.5 + (torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]) * (
            0.5 / (1 - durations[swing_idxs])
        )

        # Calculate desired contact states using von mises distribution
        smoothing_cdf_start = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf_start(foot_indices) * (
            1 - smoothing_cdf_start(foot_indices - 0.5)
        ) + smoothing_cdf_start(foot_indices - 1) * (1 - smoothing_cdf_start(foot_indices - 1.5))

        return desired_contact_states

    def _compute_force_reward(self, forces: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute force-based reward component."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:  # Negative scale means penalize unwanted contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma))
        else:  # Positive scale means reward desired contact
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(-forces[:, i] ** 2 / self.force_sigma)

        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(self, velocities: torch.Tensor, desired_contacts: torch.Tensor) -> torch.Tensor:
        """Compute velocity-based reward component."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:  # Negative scale means penalize movement during contact
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma))
        else:  # Positive scale means reward movement during swing
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)

        return (reward / velocities.shape[1]) * self.vel_scale


class GaitSymmetryReward(ManagerTermBase):
    """Penalize left-right gait asymmetry in both foot trajectories and contact states.

    The left foot at time t should match the mirrored right foot at time
    t - offset * T and vice versa, where offset comes from the gait command.
    This makes the symmetry check invariant to robot heading.

    Symmetry conditions (with commanded phase delay):
      - xz: left(t) ≈ right(t - offset * T)
      - y:  left(t) ≈ -right(t - offset * T)  (mirror-symmetric)
      - contact: left(t) ≈ right_contact(t - offset * T)
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.command_name = cfg.params["command_name"]
        self.sensor_cfg = cfg.params.get("sensor_cfg")
        self.contact_sensor: ContactSensor | None = None
        if self.sensor_cfg is not None:
            self.contact_sensor = env.scene.sensors[self.sensor_cfg.name]
        self.contact_force_threshold = cfg.params.get("contact_force_threshold", 1.0)
        self.contact_weight = cfg.params.get("contact_weight", 0.5)
        self.dt = env.step_dt

        # Circular buffer for foot positions relative to base.
        # With the current gait config, offset=0.5 and min freq=1.5 Hz,
        # the maximum commanded delay is about 0.33 s ≈ 17 steps at 0.02 s dt.
        self.max_delay = int(cfg.params.get("max_delay_steps", 50))
        # Buffer stores xyz for both feet: (num_envs, 2_feet, 3_xyz, max_delay)
        self.pos_buffer = torch.zeros(self.num_envs, 2, 3, self.max_delay, device=env.device)
        self.contact_buffer = (
            torch.zeros(self.num_envs, 2, self.max_delay, device=env.device)
            if self.contact_sensor is not None
            else None
        )
        self.buf_idx = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name,
        asset_cfg,
        sensor_cfg: SceneEntityCfg | None = None,
        contact_force_threshold: float = 1.0,
        contact_weight: float = 0.5,
        max_delay_steps=50,
    ) -> torch.Tensor:
        # Foot positions expressed in the base frame: (num_envs, 2_feet, 3_xyz)
        foot_pos_w = self.asset.data.body_link_pos_w[:, self.asset_cfg.body_ids]
        base_pos_w = self.asset.data.root_link_state_w[:, :3].unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)
        base_quat_w = self.asset.data.root_link_quat_w.unsqueeze(1).expand(-1, foot_pos_w.shape[1], -1)
        foot_rel = math_utils.quat_apply_inverse(base_quat_w, foot_pos_w - base_pos_w)

        # Clear buffer for newly reset episodes
        just_reset = self._env.episode_length_buf == 0
        if just_reset.any():
            self.pos_buffer[just_reset] = 0.0
            if self.contact_buffer is not None:
                self.contact_buffer[just_reset] = 0.0

        # Store current positions in circular buffer
        self.pos_buffer[:, :, :, self.buf_idx] = foot_rel

        # Compute commanded phase delay in steps from gait frequency and offset
        gait_params = env.command_manager.get_command(self.command_name)
        freq = gait_params[:, 0].clamp_min(1.0e-6)
        phase_offset = torch.remainder(gait_params[:, 1], 1.0)
        phase_delay_steps = (phase_offset / (freq * self.dt)).long().clamp(1, self.max_delay - 1)

        # Read delayed positions: (num_envs, 3)
        delayed_idx = (self.buf_idx - phase_delay_steps) % self.max_delay
        env_arange = torch.arange(self.num_envs, device=env.device)
        delayed_L = self.pos_buffer[env_arange, 0, :, delayed_idx]  # (num_envs, 3)
        delayed_R = self.pos_buffer[env_arange, 1, :, delayed_idx]  # (num_envs, 3)

        # Build mirror target: xz same sign, y flipped.
        # left(t) should match mirror(right(t - offset * T))
        mirror_R = delayed_R.clone()
        mirror_R[:, 1] = -mirror_R[:, 1]  # flip y
        mirror_L = delayed_L.clone()
        mirror_L[:, 1] = -mirror_L[:, 1]  # flip y

        # Position symmetry error
        position_error = (foot_rel[:, 0] - mirror_R).square().sum(dim=1) + \
                         (foot_rel[:, 1] - mirror_L).square().sum(dim=1)
        error = position_error

        # Contact symmetry error: current foot contact should match the opposite
        # foot's delayed contact state under the commanded phase offset.
        if self.contact_sensor is not None and self.contact_buffer is not None:
            contact_forces = torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
            )
            contact_state = (contact_forces > self.contact_force_threshold).float()
            self.contact_buffer[:, :, self.buf_idx] = contact_state

            delayed_contact_L = self.contact_buffer[env_arange, 0, delayed_idx]
            delayed_contact_R = self.contact_buffer[env_arange, 1, delayed_idx]
            contact_error = 0.5 * (
                torch.abs(contact_state[:, 0] - delayed_contact_R)
                + torch.abs(contact_state[:, 1] - delayed_contact_L)
            )
            error = error + self.contact_weight * contact_error

        # Only penalize once enough history is available
        valid = (self._env.episode_length_buf >= phase_delay_steps).float()

        self.buf_idx = (self.buf_idx + 1) % self.max_delay

        return error * valid


class ActionSmoothnessPenalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty


# ========================
# Jump-related rewards
# ========================


def _jump_phase_masks(jump_cmd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the staged jump command into walk / crouch / jump masks."""
    phase_signal = jump_cmd[:, 0]
    walk_phase = phase_signal < 0.25
    crouch_phase = (phase_signal >= 0.25) & (phase_signal < 0.75)
    jump_phase = phase_signal >= 0.75
    return walk_phase, crouch_phase, jump_phase


def conditional_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize z-axis linear velocity only during normal walking."""
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    walk_phase, _, _ = _jump_phase_masks(jump_cmd)
    vel_z = asset.data.root_lin_vel_w[:, 2]
    return walk_phase.float() * torch.square(vel_z)


def conditional_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    jump_scale: float = 0.2,
) -> torch.Tensor:
    """Penalize non-flat orientation. Reduced penalty during crouch/jump sequence.

    Args:
        jump_scale: multiplier on the penalty when crouching or jumping.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    _, crouch_phase, jump_phase = _jump_phase_masks(jump_cmd)
    gravity_b = asset.data.projected_gravity_b
    penalty = torch.sum(torch.square(gravity_b[:, :2]), dim=1)
    phase_active = crouch_phase | jump_phase
    scale = torch.where(phase_active, jump_scale, 1.0)
    return scale * penalty


def conditional_joint_pos_limits(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    jump_scale: float = 0.0,
) -> torch.Tensor:
    """Penalize joint position limit violations, suppressed during crouch/jump."""
    asset: Articulation = env.scene[asset_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    _, crouch_phase, jump_phase = _jump_phase_masks(jump_cmd)

    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clamp(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clamp(min=0.0)
    penalty = torch.sum(out_of_limits, dim=1)

    phase_active = crouch_phase | jump_phase
    scale = torch.where(phase_active, jump_scale, 1.0)
    return scale * penalty


def conditional_base_height(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base height deviation from target.

    When not jumping: target = commanded standing_height (cmd[:, 2]).
    When crouching/jumping: target = commanded phase_target (cmd[:, 1]).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    walk_phase, _, _ = _jump_phase_masks(jump_cmd)
    phase_target = jump_cmd[:, 1]
    standing_h = jump_cmd[:, 2]
    target = torch.where(walk_phase, standing_h, phase_target)
    return torch.abs(asset.data.root_pos_w[:, 2] - target)


def track_base_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Track standing height in walk phase and crouch target in crouch phase."""
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    walk_phase, crouch_phase, _ = _jump_phase_masks(jump_cmd)
    phase_target = jump_cmd[:, 1]
    standing_h = jump_cmd[:, 2]
    current_height = asset.data.root_pos_w[:, 2]
    walk_reward = walk_phase.float() * torch.exp(-torch.square(current_height - standing_h) / (sigma ** 2))
    crouch_reward = crouch_phase.float() * torch.exp(-torch.square(current_height - phase_target) / (sigma ** 2))
    return walk_reward + crouch_reward


def jump_upward_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Reward positive z velocity during takeoff while feet are still on ground.

    This provides the gradient signal for the robot to learn to push off the ground.
    Only active in jump phase and while at least one foot still has contact.
    Rewards clipped positive z velocity (negative vel = 0 reward).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    _, _, jump_phase = _jump_phase_masks(jump_cmd)

    # at least one foot on ground = push-off phase
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    on_ground = torch.any(forces_z > 1.0, dim=1)

    vel_z = asset.data.root_lin_vel_w[:, 2]
    # only reward upward velocity, not penalize downward
    upward_vel = torch.clamp(vel_z, min=0.0)
    return jump_phase.float() * on_ground.float() * upward_vel


def jump_flight_vel_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    velocity_command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    std: float = 0.25,
) -> torch.Tensor:
    """Track commanded xy velocity during flight phase.

    Only active in jump phase and when all feet are off the ground.
    Uses exp kernel on the xy velocity error to encourage maintaining
    forward momentum while airborne.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    vel_cmd = env.command_manager.get_command(velocity_command_name)
    _, _, jump_phase = _jump_phase_masks(jump_cmd)

    # in_flight = all feet off ground
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_flight = torch.all(forces_z < 1.0, dim=1)

    # xy velocity error
    lin_vel_error = torch.sum(
        torch.square(asset.data.root_lin_vel_b[:, :2] - vel_cmd[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / (std ** 2))

    return jump_phase.float() * in_flight.float() * reward


def jump_height_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Reward reaching the target jump height during aerial jump phase.

    Uses an exponential kernel: exp(-error^2 / sigma^2).
    Returns 0 outside the jump phase.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_flight = torch.all(forces_z < 1.0, dim=1)
    jump_cmd = env.command_manager.get_command(command_name)
    _, _, jump_phase = _jump_phase_masks(jump_cmd)
    target_height = jump_cmd[:, 1]
    current_height = asset.data.root_pos_w[:, 2]
    height_error = current_height - target_height
    reward = torch.exp(-torch.square(height_error) / (sigma ** 2))
    return jump_phase.float() * reward * in_flight.float()


def jump_landing_stability(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Reward recovering to commanded standing height after jump ends.

    When back in walk phase and at least one foot is in contact, rewards the base
    height being close to the commanded standing_height (cmd[:, 2]).
    """
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    walk_phase, _, _ = _jump_phase_masks(jump_cmd)
    standing_h = jump_cmd[:, 2]

    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    any_contact = torch.any(forces_z > 1.0, dim=1)

    height_error = asset.data.root_pos_w[:, 2] - standing_h
    reward = torch.exp(-torch.square(height_error) / (sigma ** 2))
    return walk_phase.float() * any_contact.float() * reward


def jump_tuck_legs(
    env: ManagerBasedRLEnv,
    command_name: str = "base_jump",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    tuck_angles: dict[str, float] | None = None,
    sigma: float = 0.25,
) -> torch.Tensor:
    """Reward tucking legs to target joint angles during flight phase.

    During aerial jump phase, rewards
    hip and knee joints being close to specified tuck angles.

    Args:
        tuck_angles: dict mapping joint name to target angle (rad).
        sigma: exp kernel width.
    """
    if tuck_angles is None:
        tuck_angles = {
            "hip_L_Joint": 1.15,
            "knee_L_Joint": 1.15,
            "hip_R_Joint": -1.15,
            "knee_R_Joint": -1.15,
        }

    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    jump_cmd = env.command_manager.get_command(command_name)
    _, _, jump_phase = _jump_phase_masks(jump_cmd)

    # detect flight: both wheels off ground
    forces_z = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    in_flight = torch.all(forces_z < 1.0, dim=1)

    # compute joint angle error for tuck joints
    total_error = torch.zeros(env.num_envs, device=env.device)
    for joint_name, target_angle in tuck_angles.items():
        joint_ids, _ = asset.find_joints(joint_name)
        joint_pos = asset.data.joint_pos[:, joint_ids[0]]
        total_error += torch.square(joint_pos - target_angle)

    reward = torch.exp(-total_error / (sigma ** 2))

    return jump_phase.float() * in_flight.float() * reward


def base_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Linear penalty proportional to base_Link contact force above threshold.

    penalty = max(0, |F| - threshold)

    Below threshold: 0. Above threshold: linearly increasing.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # (num_envs, n_bodies, 3)
    force_magnitude = torch.norm(forces, dim=-1).sum(dim=-1)  # (num_envs,)
    return torch.clamp(force_magnitude - threshold, min=0.0)


# ===================== terrain-adaptive helpers =====================


def _terrain_forward_gradient(
    sensor: RayCaster,
    num_x: int = 11,
    num_y: int = 11,
) -> torch.Tensor:
    """Compute forward height gradient from height scan grid.

    The grid uses "xy" ordering with x as the fastest-varying dimension.
    Positive x = forward in the robot frame (with attach_yaw_only=True).

    Returns:
        (num_envs,) tensor. >0 = uphill (stairs up), <0 = downhill, ~0 = flat.
    """
    hits_z = sensor.data.ray_hits_w[..., 2]  # (N, num_x * num_y)
    hits_z = torch.nan_to_num(hits_z, nan=0.0, posinf=0.0, neginf=0.0)
    hits_z = hits_z.view(-1, num_y, num_x)  # (N, num_y, num_x)

    mid = num_x // 2
    front = hits_z[:, :, mid + 1:]   # x > 0 (forward)
    back = hits_z[:, :, :mid]        # x < 0 (backward)

    gradient = front.mean(dim=(1, 2)) - back.mean(dim=(1, 2))
    return gradient


def _stair_edge_score(
    sensor: RayCaster,
    step_edge_threshold: float = 0.03,
    num_x: int = 11,
    num_y: int = 11,
) -> torch.Tensor:
    """Detect stairs by counting discrete height step edges in the height scan.

    Computes height differences between adjacent rays along the x (forward)
    direction. A stair edge produces a sharp height jump (e.g. 0.1-0.2m),
    while a slope has small, uniform differences.

    Args:
        sensor: RayCaster height scanner.
        step_edge_threshold: minimum height difference (m) to count as a step edge.
        num_x: grid points along x.
        num_y: grid points along y.

    Returns:
        (num_envs,) tensor. Normalized step-edge count in [0, 1].
        0 = no edges (flat/slope), 1 = many edges (stairs).
    """
    hits_z = sensor.data.ray_hits_w[..., 2]  # (N, num_x * num_y)
    hits_z = torch.nan_to_num(hits_z, nan=0.0, posinf=0.0, neginf=0.0)
    hits_z = hits_z.view(-1, num_y, num_x)  # (N, num_y, num_x)

    # height difference between adjacent points along x (forward direction)
    diffs = torch.abs(hits_z[:, :, 1:] - hits_z[:, :, :-1])  # (N, num_y, num_x-1)
    # count jumps exceeding threshold
    step_count = (diffs > step_edge_threshold).float().sum(dim=(1, 2))  # (N,)
    # normalize: max possible edges = num_y * (num_x - 1)
    max_edges = num_y * (num_x - 1)
    return torch.clamp(step_count / max_edges, 0.0, 1.0)


def _uphill_stairs_score(
    sensor: RayCaster,
    gradient_threshold: float = 0.02,
    step_edge_threshold: float = 0.03,
    num_x: int = 11,
    num_y: int = 11,
) -> torch.Tensor:
    """Compute [0, 1] uphill-stairs score: 1 when climbing stairs, 0 on flat/slope/downhill.

    Combines two signals:
      - Forward gradient: front rays higher than back rays (uphill)
      - Step edges: discrete height jumps between adjacent rays (stairs, not slope)

    Only when BOTH uphill AND step edges are present does the score approach 1.
    Smooth slopes have few step edges → score ≈ 0.
    """
    gradient = _terrain_forward_gradient(sensor, num_x, num_y)
    stair_edges = _stair_edge_score(sensor, step_edge_threshold, num_x, num_y)
    # only positive gradient = uphill; clamp negatives to 0
    uphill = torch.clamp(gradient / gradient_threshold, 0.0, 1.0)
    # multiply: need both uphill AND step edges to trigger
    return uphill * stair_edges


# ===================== terrain-adaptive rewards =====================


def _gait_needed_score(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    gradient_threshold: float = 0.02,
    step_edge_threshold: float = 0.03,
    lateral_vel_threshold: float = 0.1,
) -> torch.Tensor:
    """Compute [0, 1] score indicating whether gait (legged locomotion) is needed.

    Gait is needed when either:
      1. Climbing upstairs (uphill + step edges detected)
      2. Lateral velocity (Vy) is commanded (wheels can't move sideways)

    Returns the max of the two signals, clamped to [0, 1].
    """
    # --- stair detection ---
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    stairs = _uphill_stairs_score(sensor, gradient_threshold, step_edge_threshold)

    # --- lateral velocity command ---
    vel_cmd = env.command_manager.get_command(command_name)
    vy = torch.abs(vel_cmd[:, 1])  # lateral velocity command
    lateral = torch.clamp(vy / lateral_vel_threshold, 0.0, 1.0)

    # either condition triggers gait
    return torch.max(stairs, lateral)


def terrain_adaptive_wheel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    gradient_threshold: float = 0.02,
    step_edge_threshold: float = 0.03,
    lateral_vel_threshold: float = 0.1,
) -> torch.Tensor:
    """Extra penalty on wheel velocity when gait is needed (stairs or lateral movement).

    On flat ground with no lateral command the penalty is zero.
    """
    gait_score = _gait_needed_score(
        env, sensor_cfg, command_name, gradient_threshold, step_edge_threshold, lateral_vel_threshold
    )
    asset: Articulation = env.scene[asset_cfg.name]
    wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return gait_score * torch.sum(torch.square(wheel_vel), dim=1)


def terrain_adaptive_gait_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    gradient_threshold: float = 0.02,
    step_edge_threshold: float = 0.03,
    lateral_vel_threshold: float = 0.1,
    air_time_threshold: float = 0.5,
) -> torch.Tensor:
    """Reward biped gait (alternating single stance) when gait is needed.

    Uses feet_air_time_positive_biped-style reward:
      - Rewards single-stance phases (one foot in air, one on ground)
      - Clamped by air_time_threshold

    Gated by gait_needed_score: only active on stairs or lateral movement.
    """
    gait_score = _gait_needed_score(
        env, sensor_cfg, command_name, gradient_threshold, step_edge_threshold, lateral_vel_threshold
    )

    # biped gait reward: single stance with alternating feet
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, contact_sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=air_time_threshold)

    return gait_score * reward
