"""Sub-module containing command generator for jump tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import JumpCommandCfg


class JumpCommand(CommandTerm):
    """Command generator with explicit crouch and jump phases.

    Output shape: (num_envs, 3)
        [0] phase_signal:       0.0 (walk), 0.5 (crouch), 1.0 (takeoff/flight)
        [1] phase_target:       crouch target during crouch, jump peak target during jump
        [2] standing_height:    commanded base height for normal locomotion / landing

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
    WALK_PHASE: float = 0.0
    CROUCH_PHASE: float = 0.5
    JUMP_PHASE: float = 1.0

    def __init__(self, cfg: JumpCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # command buffer: [phase_signal, phase_target, standing_height]
        self.jump_cmd = torch.zeros(self.num_envs, 3, device=self.device)
        mid = (cfg.standing_height_range[0] + cfg.standing_height_range[1]) / 2.0
        self.jump_cmd[:, 2] = mid

        # stored peak target for the pending / active jump
        self._pending_jump_target = torch.zeros(self.num_envs, device=self.device)

        # jump window countdown — used as max timeout safety net
        self._time_remaining = torch.zeros(self.num_envs, device=self.device)

        # assist-force countdown per env
        self._assist_remaining = torch.zeros(self.num_envs, device=self.device)

        # contact-based landing detection: track previous step flight state
        self._prev_in_flight = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # resolve contact sensor body IDs for landing detection
        from isaaclab.sensors import ContactSensor
        contact_sensor: ContactSensor = env.scene.sensors[cfg.contact_sensor_name]
        self._contact_sensor = contact_sensor
        robot = env.scene["robot"]
        self._contact_body_ids, _ = robot.find_bodies(cfg.contact_body_names)

        # resolve base body index once at init
        self._assist_body_id: int | None = None
        if cfg.assist_force_max > 0.0:
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

    def _is_crouch_phase(self) -> torch.Tensor:
        return (self.jump_cmd[:, 0] > 0.25) & (self.jump_cmd[:, 0] < 0.75)

    def _is_jump_phase(self) -> torch.Tensor:
        return self.jump_cmd[:, 0] >= 0.75

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
            active_env_ids = env_ids[jump_ids]
            self.jump_cmd[active_env_ids, 0] = self.CROUCH_PHASE
            self.jump_cmd[active_env_ids, 1] = self.cfg.crouch_height
            self._pending_jump_target[active_env_ids] = target_h
            self._time_remaining[active_env_ids] = 0.0
            self._assist_remaining[active_env_ids] = 0.0
            self._prev_in_flight[active_env_ids] = False

        walk_ids = (~trigger_mask).nonzero(as_tuple=False).squeeze(-1)
        if walk_ids.numel() > 0:
            inactive_env_ids = env_ids[walk_ids]
            self.jump_cmd[inactive_env_ids, 0] = self.WALK_PHASE
            self.jump_cmd[inactive_env_ids, 1] = 0.0
            self._pending_jump_target[inactive_env_ids] = 0.0
            self._time_remaining[inactive_env_ids] = 0.0
            self._assist_remaining[inactive_env_ids] = 0.0
            self._prev_in_flight[inactive_env_ids] = False

    def _update_command(self):
        """Progress crouch -> jump -> landing and apply assist force after takeoff."""
        # --- contact-based landing detection ---
        forces_z = self._contact_sensor.data.net_forces_w[:, self._contact_body_ids, 2]
        in_flight = torch.all(forces_z < self.cfg.contact_force_threshold, dim=1)
        any_contact = torch.any(forces_z > self.cfg.contact_force_threshold, dim=1)
        robot = self._env.scene["robot"]
        current_height = robot.data.root_pos_w[:, 2]

        # landing = was in flight last step, now has contact
        landed = self._prev_in_flight & any_contact
        self._prev_in_flight = in_flight

        # --- crouch-to-takeoff transition ---
        crouch_mask = self._is_crouch_phase()
        if crouch_mask.any():
            crouch_error = torch.abs(current_height - self.jump_cmd[:, 1])
            crouch_ready = crouch_mask & any_contact & (crouch_error <= self.cfg.crouch_tolerance)
            if crouch_ready.any():
                self.jump_cmd[crouch_ready, 0] = self.JUMP_PHASE
                self.jump_cmd[crouch_ready, 1] = self._pending_jump_target[crouch_ready]
                self._time_remaining[crouch_ready] = self._compute_jump_duration(
                    current_height[crouch_ready], self._pending_jump_target[crouch_ready]
                )
                self._assist_remaining[crouch_ready] = self.cfg.assist_force_duration

        # --- jump window countdown (max timeout safety net) ---
        active_mask = self._is_jump_phase()
        if active_mask.any():
            self._time_remaining[active_mask] -= self._env.step_dt
            expired = active_mask & ((self._time_remaining <= 0.0) | landed)
            self.jump_cmd[expired, 0] = 0.0
            self.jump_cmd[expired, 1] = 0.0
            self._pending_jump_target[expired] = 0.0
            self._time_remaining[expired] = 0.0
            self._assist_remaining[expired] = 0.0

        # --- assist force ---
        if self._assist_body_id is None:
            return
        assist_mask = self._assist_remaining > 0.0
        robot.permanent_wrench_composer.reset()

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
        msg += f"\tCrouch height: {self.cfg.crouch_height}\n"
        msg += f"\tCrouch tolerance: {self.cfg.crouch_tolerance}\n"
        msg += f"\tJump delta range: {self.cfg.jump_delta_range}\n"
        msg += f"\tStanding height range: {self.cfg.standing_height_range}\n"
        msg += f"\tAssist force max: {self.cfg.assist_force_max} N\n"
        msg += f"\tAssist duration: {self.cfg.assist_force_duration} s\n"
        return msg

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
