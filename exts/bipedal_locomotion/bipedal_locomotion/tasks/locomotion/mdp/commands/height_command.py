"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import HeightCommandCfg

class HeightCommand(CommandTerm):
    """Command generator that generates gait frequency, phase offset and contact duration."""

    cfg: HeightCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: HeightCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)
        # command tensor
        self.height_command = torch.full((self.num_envs, 1), (self.cfg.max_height + self.cfg.min_height) / 2, device=self.device)
        # create metrics dictionary for logging
        self.metrics = {}

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "HeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The height command. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        """Update the metrics based on the current state.

        In this implementation, we don't track any specific metrics.
        """
        pass

    def _resample_command(self, env_ids):
        """Resample the height command for specified environments."""
        # sample height parameters
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(self.cfg.min_height, self.cfg.max_height)

    def _update_command(self):
        """Update the command. No additional processing needed in this implementation."""
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        In this implementation, we don't provide any debug visualization.
        """
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        In this implementation, we don't provide any debug visualization.
        """
        pass
