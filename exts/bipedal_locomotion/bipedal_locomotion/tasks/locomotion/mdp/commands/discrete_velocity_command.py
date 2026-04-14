from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
    from .commands_cfg import DiscreteLevelVelocityCommandCfg


class DiscreteVelocityCommand(UniformVelocityCommand):
    """Velocity command that samples linear-x from a discrete candidate set."""

    cfg: "DiscreteLevelVelocityCommandCfg"

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        choices = torch.tensor(self.cfg.ranges.lin_vel_x_choices, device=self.device, dtype=self.vel_command_b.dtype)
        if choices.numel() == 0:
            raise ValueError("lin_vel_x_choices must contain at least one discrete velocity.")

        r = torch.empty(len(env_ids), device=self.device)
        choice_ids = torch.randint(0, choices.numel(), (len(env_ids),), device=self.device)

        self.vel_command_b[env_ids, 0] = choices[choice_ids]
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
