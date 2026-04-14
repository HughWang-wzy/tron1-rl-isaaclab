import math
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .active_group_velocity_command import ActiveGroupVelocityCommand
from .discrete_velocity_command import DiscreteVelocityCommand
from .gait_command import GaitCommand  # Import the GaitCommand class
from .height_command import HeightCommand  # Import the HeightCommand class
from .jump_command import JumpCommand  # Import the JumpCommand class


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Velocity command with limit_ranges for curriculum-based range expansion."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Hard limits that the curriculum cannot exceed."""


@configclass
class DiscreteLevelVelocityCommandCfg(UniformLevelVelocityCommandCfg):
    """Velocity command with discrete linear-x choices."""

    class_type: type = DiscreteVelocityCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        lin_vel_x_choices: tuple[float, ...] = MISSING
        """Discrete candidate values for the linear-x velocity command."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class ActiveGroupVelocityCommandCfg(CommandTermCfg):
    """Debug-only velocity command that visualizes the active expert velocity."""

    class_type: type = ActiveGroupVelocityCommand

    asset_name: str = MISSING
    jump_velocity_command_name: str = MISSING
    gait_velocity_command_name: str = MISSING
    num_groups: int = 2
    jump_group_id: int = 0
    # This term does not sample its own command, but CommandTerm.reset() still
    # draws from resampling_time_range, so it must stay finite.
    resampling_time_range: tuple[float, float] = (1.0, 1.0)

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


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
    crouch_height: float = 0.7
    """Target base height [m] for the pre-jump crouch phase."""
    crouch_tolerance: float = 0.02
    """Absolute height error tolerance [m] required to start takeoff."""
    jump_margin: float = 0.5
    """Extra time (seconds) added to ballistic flight time for crouch + landing."""
    resampling_time_range: tuple[float, float] = (3.0, 10.0)
    """Time interval for resampling the command (in seconds)."""

    # ------------------------------------------------------------------
    # Assist-force (magnitude updated each iteration by curriculum term)
    # ------------------------------------------------------------------
    assist_force_max: float = 1000.0
    """Upward assist force [N] at jump trigger. Updated each iter by curriculum. Set 0 to disable."""
    assist_force_duration: float = 0.5
    """How long [s] the assist force is applied after each jump trigger."""
    assist_body_name: str = "base_Link"
    """Name of the robot body to apply the assist force to."""

    # ------------------------------------------------------------------
    # Contact-based landing detection
    # ------------------------------------------------------------------
    contact_sensor_name: str = "contact_forces"
    """Name of the contact sensor in the scene."""
    contact_body_names: list[str] | str = "wheel_.*"
    """Body names for feet/wheels used to detect ground contact."""
    contact_force_threshold: float = 1.0
    """Force threshold [N] above which a body is considered in contact."""
