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
