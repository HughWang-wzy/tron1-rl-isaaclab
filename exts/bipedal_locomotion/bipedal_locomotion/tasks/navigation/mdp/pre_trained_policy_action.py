"""Pre-trained policy action term for WheelFoot robot.

Adapted from IsaacLab's PreTrainedPolicyAction to support multiple low-level
action terms (joint_pos + joint_vel), as required by the WF robot.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class WFPreTrainedPolicyAction(ActionTerm):
    """Pre-trained policy action term for WheelFoot robot.

    This action term loads a pre-trained locomotion policy and applies
    low-level joint actions. The high-level raw actions are 3D velocity
    commands (vx, vy, yaw_rate) that get fed to the low-level policy
    as velocity commands.

    Supports two low-level action terms: joint_pos and joint_vel.
    """

    cfg: WFPreTrainedPolicyActionCfg

    def __init__(self, cfg: WFPreTrainedPolicyActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]

        # load pre-trained policy
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        file_bytes = read_file(cfg.policy_path)
        self.policy = torch.jit.load(file_bytes).to(env.device).eval()

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)

        # prepare low-level action terms
        self._ll_joint_pos_term: ActionTerm = cfg.low_level_joint_pos.class_type(cfg.low_level_joint_pos, env)
        self._ll_joint_vel_term: ActionTerm = cfg.low_level_joint_vel.class_type(cfg.low_level_joint_vel, env)

        ll_pos_dim = self._ll_joint_pos_term.action_dim
        ll_vel_dim = self._ll_joint_vel_term.action_dim
        self._ll_total_action_dim = ll_pos_dim + ll_vel_dim
        self._ll_pos_dim = ll_pos_dim

        self.low_level_actions = torch.zeros(self.num_envs, self._ll_total_action_dim, device=self.device)

        def last_action():
            if hasattr(env, "episode_length_buf"):
                self.low_level_actions[env.episode_length_buf == 0, :] = 0
            return self.low_level_actions

        # remap low-level observations: actions and velocity_commands
        cfg.low_level_observations.actions.func = lambda dummy_env: last_action()
        cfg.low_level_observations.actions.params = dict()
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self._raw_actions
        cfg.low_level_observations.velocity_commands.params = dict()

        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

        self._counter = 0

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
            self.low_level_actions[:] = self.policy(low_level_obs)
            # split actions into joint_pos and joint_vel
            pos_actions = self.low_level_actions[:, : self._ll_pos_dim]
            vel_actions = self.low_level_actions[:, self._ll_pos_dim :]
            self._ll_joint_pos_term.process_actions(pos_actions)
            self._ll_joint_vel_term.process_actions(vel_actions)
            self._counter = 0
        self._ll_joint_pos_term.apply_actions()
        self._ll_joint_vel_term.apply_actions()
        self._counter += 1

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "base_vel_goal_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.raw_actions[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class WFPreTrainedPolicyActionCfg(ActionTermCfg):
    """Configuration for WF pre-trained policy action term."""

    class_type: type[ActionTerm] = WFPreTrainedPolicyAction
    asset_name: str = MISSING
    """Name of the robot asset in the scene."""
    policy_path: str = MISSING
    """Path to the pre-trained locomotion policy (.pt file)."""
    low_level_decimation: int = 4
    """Decimation factor for the low-level action term."""
    low_level_joint_pos: ActionTermCfg = MISSING
    """Low-level joint position action configuration."""
    low_level_joint_vel: ActionTermCfg = MISSING
    """Low-level joint velocity action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low-level observation configuration."""
    debug_vis: bool = True
