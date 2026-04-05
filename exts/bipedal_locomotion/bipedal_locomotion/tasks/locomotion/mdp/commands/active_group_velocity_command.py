"""Debug-only command term for visualizing the active expert velocity."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import ActiveGroupVelocityCommandCfg


class ActiveGroupVelocityCommand(CommandTerm):
    """Visualize the velocity command currently active for each expert group."""

    cfg: ActiveGroupVelocityCommandCfg

    def __init__(self, cfg: ActiveGroupVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.active_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics = {}

        if cfg.jump_group_id < 0 or cfg.jump_group_id >= cfg.num_groups:
            raise ValueError(
                f"jump_group_id must be within [0, {cfg.num_groups - 1}], got {cfg.jump_group_id}."
            )

    @property
    def command(self) -> torch.Tensor:
        """The active expert velocity command. Shape is (num_envs, 3)."""
        return self.active_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        del env_ids

    def _update_command(self):
        jump_velocity = self._env.command_manager.get_command(self.cfg.jump_velocity_command_name)
        gait_velocity = self._env.command_manager.get_command(self.cfg.gait_velocity_command_name)
        group_ids = torch.arange(self.num_envs, device=self.device) * self.cfg.num_groups // self.num_envs
        self.active_command_b[:] = torch.where(
            group_ids.unsqueeze(-1) == self.cfg.jump_group_id,
            jump_velocity,
            gait_velocity,
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        del event
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat
