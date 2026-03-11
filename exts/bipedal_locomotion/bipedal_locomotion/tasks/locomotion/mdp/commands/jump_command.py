"""Sub-module containing command generator for jump tasks."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import JumpCommandCfg


class JumpCommand(CommandTerm):
    """Command generator that produces jump trigger, target height, and standing height.

    Output shape: (num_envs, 3)
        [0] jump_active:      0.0 (walk) or 1.0 (jump)
        [1] target_jump_height: desired peak base height during jump (meters)
        [2] standing_height:   commanded base height for normal locomotion (meters)

    standing_height is always sampled (even when not jumping) so the robot
    learns to control its base height during normal walking.

    When jump_active=1, the robot should reach target_jump_height then return
    to standing_height.  The jump window duration is derived from the height
    delta using ballistic physics.
    """

    cfg: JumpCommandCfg

    GRAVITY: float = 9.81

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # command buffer: [jump_active, target_jump_height, standing_height]
        self.jump_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        # initialise standing_height to mid-range
        mid = (self.cfg.standing_height_range[0] + self.cfg.standing_height_range[1]) / 2.0
        self.jump_cmd[:, 2] = mid
        # internal countdown (not exposed to policy)
        self._time_remaining = torch.zeros(self.num_envs, device=self.device)
        self.metrics = {}

    def __str__(self) -> str:
        msg = "JumpCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tJump probability: {self.cfg.jump_probability}\n"
        msg += f"\tJump delta range: {self.cfg.jump_delta_range}\n"
        msg += f"\tStanding height range: {self.cfg.standing_height_range}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The jump command. Shape is (num_envs, 3)."""
        return self.jump_cmd

    def _compute_jump_duration(self, standing_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
        """Derive jump window duration from height delta using ballistic model.

        total_time = flight_time + margin
        flight_time = 2 * sqrt(2 * delta_h / g)
        """
        delta_h = (target_h - standing_h).clamp(min=0.01)
        flight_time = 2.0 * torch.sqrt(2.0 * delta_h / self.GRAVITY)
        return flight_time + self.cfg.jump_margin

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return

        # always resample standing_height for all resampled envs
        standing_h = torch.empty(n, device=self.device).uniform_(
            self.cfg.standing_height_range[0], self.cfg.standing_height_range[1]
        )
        self.jump_cmd[env_ids, 2] = standing_h

        # decide which envs get a jump trigger
        trigger_mask = torch.rand(n, device=self.device) < self.cfg.jump_probability

        # jump envs: target = standing_height + delta
        jump_ids = trigger_mask.nonzero(as_tuple=False).squeeze(-1)
        if jump_ids.numel() > 0:
            delta = torch.empty(jump_ids.numel(), device=self.device).uniform_(
                self.cfg.jump_delta_range[0], self.cfg.jump_delta_range[1]
            )
            jump_standing = standing_h[jump_ids]
            target_h = jump_standing + delta
            self.jump_cmd[env_ids[jump_ids], 0] = 1.0
            self.jump_cmd[env_ids[jump_ids], 1] = target_h
            self._time_remaining[env_ids[jump_ids]] = self._compute_jump_duration(jump_standing, target_h)

        # walk envs: no jump
        walk_ids = (~trigger_mask).nonzero(as_tuple=False).squeeze(-1)
        if walk_ids.numel() > 0:
            self.jump_cmd[env_ids[walk_ids], 0] = 0.0
            self.jump_cmd[env_ids[walk_ids], 1] = 0.0
            self._time_remaining[env_ids[walk_ids]] = 0.0

    def _update_command(self):
        """Countdown internal timer for active jumps; deactivate when expired."""
        active_mask = self.jump_cmd[:, 0] > 0.5
        if active_mask.any():
            self._time_remaining[active_mask] -= self._env.step_dt
            expired = active_mask & (self._time_remaining <= 0.0)
            self.jump_cmd[expired, 0] = 0.0
            self.jump_cmd[expired, 1] = 0.0
            self._time_remaining[expired] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
