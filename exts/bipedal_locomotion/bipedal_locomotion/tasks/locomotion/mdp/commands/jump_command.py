"""Sub-module containing command generator for jump tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import JumpCommandCfg


class JumpCommand(CommandTerm):
    """Command generator that produces jump trigger, target height, and standing height.

    Output shape: (num_envs, 3)
        [0] jump_active:        0.0 (walk) or 1.0 (jump)
        [1] target_jump_height: desired peak base height during jump (meters)
        [2] standing_height:    commanded base height for normal locomotion (meters)

    standing_height is always sampled (even when not jumping) so the robot
    learns to control its base height during normal walking.

    Optionally applies an upward assist force when a jump is triggered
    to bootstrap the learning signal.  The force is applied for
    ``cfg.assist_force_duration`` seconds at the start of each jump.
    Its magnitude ``cfg.assist_force_max`` is updated externally by a
    curriculum term (see ``jump_assist_force_curriculum``).  Set
    ``assist_force_max = 0`` to disable the feature entirely.
    """

    cfg: JumpCommandCfg

    GRAVITY: float = 9.81

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # command buffer: [jump_active, target_jump_height, standing_height]
        self.jump_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        mid = (cfg.standing_height_range[0] + cfg.standing_height_range[1]) / 2.0
        self.jump_cmd[:, 2] = mid

        # jump window countdown (not exposed to policy)
        self._time_remaining = torch.zeros(self.num_envs, device=self.device)

        # assist-force countdown per env
        self._assist_remaining = torch.zeros(self.num_envs, device=self.device)

        # resolve base body index once at init
        self._assist_body_id: int | None = None
        if cfg.assist_force_max > 0.0:
            robot = env.scene["robot"]
            body_ids, _ = robot.find_bodies(cfg.assist_body_name)
            self._assist_body_id = body_ids[0]

        self.metrics = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def command(self) -> torch.Tensor:
        """The jump command. Shape is (num_envs, 3)."""
        return self.jump_cmd

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_jump_duration(self, standing_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
        """Ballistic flight time + margin."""
        delta_h = (target_h - standing_h).clamp(min=0.01)
        flight_time = 2.0 * torch.sqrt(2.0 * delta_h / self.GRAVITY)
        return flight_time + self.cfg.jump_margin

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return

        # always resample standing_height
        standing_h = torch.empty(n, device=self.device).uniform_(
            self.cfg.standing_height_range[0], self.cfg.standing_height_range[1]
        )
        self.jump_cmd[env_ids, 2] = standing_h

        # decide which envs jump this interval
        trigger_mask = torch.rand(n, device=self.device) < self.cfg.jump_probability

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
            # start assist-force countdown
            self._assist_remaining[env_ids[jump_ids]] = self.cfg.assist_force_duration

        walk_ids = (~trigger_mask).nonzero(as_tuple=False).squeeze(-1)
        if walk_ids.numel() > 0:
            self.jump_cmd[env_ids[walk_ids], 0] = 0.0
            self.jump_cmd[env_ids[walk_ids], 1] = 0.0
            self._time_remaining[env_ids[walk_ids]] = 0.0
            self._assist_remaining[env_ids[walk_ids]] = 0.0

    def _update_command(self):
        """Countdown timers and apply assist force at current cfg magnitude."""
        # --- jump window countdown ---
        active_mask = self.jump_cmd[:, 0] > 0.5
        if active_mask.any():
            self._time_remaining[active_mask] -= self._env.step_dt
            expired = active_mask & (self._time_remaining <= 0.0)
            self.jump_cmd[expired, 0] = 0.0
            self.jump_cmd[expired, 1] = 0.0
            self._time_remaining[expired] = 0.0
        # print(f"command: {self.jump_cmd}")  # for debugging
        # print(f"time remaining: {self._time_remaining}")  # for debugging
        # --- assist force ---
        if self._assist_body_id is None:
            return

        assist_mask = self._assist_remaining > 0.0
        robot = self._env.scene["robot"]

        # forces shape: (num_envs, 1, 3) — only the base body
        forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        torques = torch.zeros_like(forces)

        if assist_mask.any():
            force_mag = self.cfg.assist_force_max  # updated each iter by curriculum
            if force_mag > 0.0:
                forces[assist_mask, 0, 2] = force_mag  # upward Z
            self._assist_remaining[assist_mask] -= self._env.step_dt
            self._assist_remaining.clamp_(min=0.0)
        robot.permanent_wrench_composer.set_forces_and_torques(
            forces, torques, body_ids=[self._assist_body_id], is_global=True
        )

    def __str__(self) -> str:
        msg = "JumpCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tJump probability: {self.cfg.jump_probability}\n"
        msg += f"\tJump delta range: {self.cfg.jump_delta_range}\n"
        msg += f"\tStanding height range: {self.cfg.standing_height_range}\n"
        msg += f"\tAssist force max: {self.cfg.assist_force_max} N\n"
        msg += f"\tAssist duration: {self.cfg.assist_force_duration} s\n"
        return msg

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
