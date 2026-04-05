"""Custom termination functions for bipedal locomotion tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_contact_and_bad_orientation(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    limit_angle: float,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate only when base contact AND bad orientation occur simultaneously.

    This avoids premature termination during jumping (where base contact alone
    may happen during landing) by requiring the robot to also be in a bad
    orientation (e.g. fallen over).

    Args:
        sensor_cfg: Contact sensor config with body_names for base link.
        limit_angle: Max allowable tilt angle (radians) from vertical.
        threshold: Contact force threshold (N).
        asset_cfg: Robot asset config.
    """
    # --- base contact ---
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    has_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )

    # --- bad orientation ---
    asset: RigidObject = env.scene[asset_cfg.name]
    has_bad_orient = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle

    return torch.logical_and(has_contact, has_bad_orient)


def base_contact_and_bad_orientation_after_grace(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    limit_angle: float,
    threshold: float = 1.0,
    grace_steps: int = 30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate after sustained base contact with bad orientation.

    Compared with the immediate termination variant, this gives the policy a
    short recovery window to roll or push itself back upright before the
    episode ends.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    has_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )

    asset: RigidObject = env.scene[asset_cfg.name]
    has_bad_orient = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
    fallen = torch.logical_and(has_contact, has_bad_orient)

    counter_name = f"_{sensor_cfg.name}_{asset_cfg.name}_fallen_steps"
    counter = getattr(env, counter_name, None)
    if counter is None or counter.shape[0] != env.num_envs or counter.device != env.device:
        counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        setattr(env, counter_name, counter)

    counter[fallen] += 1
    counter[~fallen] = 0

    return fallen & (counter >= grace_steps)


def base_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot base height drops below a minimum threshold."""

    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < minimum_height
