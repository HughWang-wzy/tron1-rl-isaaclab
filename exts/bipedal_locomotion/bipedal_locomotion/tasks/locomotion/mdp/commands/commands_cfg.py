import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .gait_command import GaitCommand  # Import the GaitCommand class
from .height_command import HeightCommand  # Import the HeightCommand class
from .jump_command import JumpCommand  # Import the JumpCommand class


@configclass
class UniformGaitCommandCfg(CommandTermCfg):
    """Configuration for the gait command generator."""

    class_type: type = GaitCommand  # Specify the class type for dynamic instantiation

    @configclass
    class Ranges:
        """Uniform distribution ranges for the gait parameters."""

        frequencies: tuple[float, float] = MISSING
        """Range for gait frequencies [Hz]."""
        offsets: tuple[float, float] = MISSING
        """Range for phase offsets [0-1]."""
        durations: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""
        swing_height: tuple[float, float] = MISSING
        """Range for contact durations [0-1]."""

    ranges: Ranges = MISSING
    """Distribution ranges for the gait parameters."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the gait (in seconds)."""

@configclass
class HeightCommandCfg(CommandTermCfg):
    """Configuration for the height command generator."""

    class_type: type = HeightCommand  # Specify the class type for dynamic instantiation

    min_height: float = MISSING
    """Minimum height [m]."""
    max_height: float = MISSING
    """Maximum height [m]."""

    resampling_time_range: tuple[float, float] = MISSING
    """Time interval for resampling the height (in seconds)."""


@configclass
class JumpCommandCfg(CommandTermCfg):
    """Configuration for the jump command generator."""

    class_type: type = JumpCommand

    jump_probability: float = 0.15
    """Probability of triggering a jump at each resample."""
    standing_height_range: tuple[float, float] = (0.6, 0.9)
    """Range for the commanded standing base height [m]. Sampled every resample."""
    jump_delta_range: tuple[float, float] = (0.05, 0.2)
    """Range for the jump height delta above standing_height [m]."""
    jump_margin: float = 0.5
    """Extra time (seconds) added to ballistic flight time for crouch + landing."""
    resampling_time_range: tuple[float, float] = (3.0, 10.0)
    """Time interval for resampling the command (in seconds)."""