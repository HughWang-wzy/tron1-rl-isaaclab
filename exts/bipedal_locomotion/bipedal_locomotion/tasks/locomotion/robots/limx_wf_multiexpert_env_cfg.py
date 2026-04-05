"""Multi-expert distillation student environment: Jump + Gait.

"""

import math

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFJumpFlatEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.WF.limx_base_env_cfg import ObservarionsCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

_BASE_JUMP_STANDING_HEIGHT_INDEX = 2
_JUMP_VELOCITY_COMMAND_NAME = "base_velocity_jump"
_GAIT_VELOCITY_COMMAND_NAME = "base_velocity_gait"
_JUMP_COMMAND_NAME = "base_jump"


# ---------------------------------------------------------------------------
# Flattened history obs group (encoder input for teacher JITs)
# ---------------------------------------------------------------------------

@configclass
class _HistoryObsFlatCfg(ObsGroup):
    """Proprioceptive history with flatten_history_dim=True.

    Shape: (num_envs, history_length * obs_dim).
    Matches the input expected by the encoder when it is JIT-wrapped together
    with the actor for use as a frozen teacher.
    """

    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        clip=(-100.0, 100.0),
        scale=0.25,
    )
    proj_gravity = ObsTerm(
        func=mdp.projected_gravity,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel_exclude_wheel,
        params={"wheel_joints_name": ["wheel_[RL]_Joint"]},
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        clip=(-100.0, 100.0),
        scale=0.05,
    )
    last_action = ObsTerm(
        func=mdp.last_action,
        clip=(-100.0, 100.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 10
        self.flatten_history_dim = False


# ---------------------------------------------------------------------------
# Combined multi-expert student environment
# ---------------------------------------------------------------------------

@configclass
class WFMultiExpertFlatEnvCfg(WFJumpFlatEnvCfg):
    """Student environment for Jump + Gait multi-expert distillation.
    """

    def __post_init__(self):
        super().__post_init__()   # full Jump flat env setup

        # Disable all curriculum terms for multi-expert distillation runs.
        self.curriculum = None
        self.rewards = None
        # Remove the inherited single-expert velocity command to avoid misleading
        # debug markers; multi-expert uses the separated jump/gait velocity terms below.
        self.commands.base_velocity = None
        self.events.reset_fallen_robot = EventTerm(
            func=mdp.reset_robot_fallen_state_for_env_group,
            mode="reset",
            params={
                "probability": 0.03,
                "target_group": 0,
                "num_groups": 2,
                "base_height_range": (0.22, 0.34),
                "pitch_range": (1.35, 1.75),
                "roll_range": (1.35, 1.75),
                "yaw_range": (-math.pi, math.pi),
                "xy_range": (-0.20, 0.20),
                "velocity_range": {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "z": (-0.1, 0.1),
                    "roll": (-0.2, 0.2),
                    "pitch": (-0.2, 0.2),
                    "yaw": (-0.2, 0.2),
                },
                "joint_position_noise_range": (-0.04, 0.04),
            },
        )
        self.commands.base_jump = mdp.JumpCommandCfg(
            jump_probability=0.5,
            standing_height_range=(0.7, 0.9),
            jump_delta_range=(0.25, 0.5),
            jump_margin=0.5,
            resampling_time_range=(3.0, 10.0),
        )
        self.commands.base_jump.assist_force_max = 0.0
        self.events.push_robot = EventTerm(
            func=mdp.apply_external_force_torque_stochastic,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
                "force_range": {
                    "x": (-500.0, 500.0),
                    "y": (-500.0, 500.0),
                    "z": (-0.0, 0.0),
                },  # force = mass * dv / dt
                "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
                "probability": 0.002,  # Expect step = 1 / probability
            },
        )
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 1.0},
        )
        # ===================== separated velocity commands =====================
        self.commands.base_velocity_jump = mdp.UniformLevelVelocityCommandCfg(
            asset_name="robot",
            heading_command=True,
            heading_control_stiffness=1.0,
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            debug_vis=False,
            resampling_time_range=(10.0, 10.0),
            ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-4.0, 4.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-math.pi, math.pi),
                heading=(-math.pi, math.pi),
            ),
            limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-4.0, 4.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-math.pi, math.pi),
                heading=(-math.pi, math.pi),
            ),
        )
        self.commands.base_velocity_gait = mdp.UniformLevelVelocityCommandCfg(
            asset_name="robot",
            heading_command=True,
            heading_control_stiffness=1.0,
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            debug_vis=False,
            resampling_time_range=(10.0, 10.0),
            ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-math.pi / 2, math.pi / 2),
                heading=(-math.pi, math.pi),
            ),
            limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-math.pi / 2, math.pi / 2),
                heading=(-math.pi, math.pi),
            ),
        )
        self.commands.base_velocity_active_debug = mdp.ActiveGroupVelocityCommandCfg(
            asset_name="robot",
            jump_velocity_command_name=_JUMP_VELOCITY_COMMAND_NAME,
            gait_velocity_command_name=_GAIT_VELOCITY_COMMAND_NAME,
            num_groups=2,
            jump_group_id=0,
            debug_vis=False,
        )

        # ===================== gait commands =====================
        self.commands.gait_command = mdp.UniformGaitCommandCfg(
            resampling_time_range=(5.0, 5.0),
            debug_vis=False,
            ranges=mdp.UniformGaitCommandCfg.Ranges(
                frequencies=(1.5, 2.5),
                offsets=(0.5, 0.5),
                durations=(0.5, 0.5),
                swing_height=(0.1, 0.2),
            ),
        )

        # ===================== env_group obs =====================
        # Shape (N, 1).  0.0 = jump expert,  1.0 = gait expert.
        self.observations.env_group = ObservarionsCfg.ExpertTargetCfg()
        self.observations.env_group.group_id = ObsTerm(
            func=mdp.env_group_id,
            params={"num_groups": 2},
        )

        # ===================== commands obs group =====================
        # Keep shared ``commands`` as velocity-only dim=3 while sampling jump/gait
        # velocity commands from separate command terms by env-group.
        self.observations.commands = ObservarionsCfg.ExpertTargetCfg()
        self.observations.commands.expert_commands = ObsTerm(
            func=mdp.expert_separated_commands,
            params={
                "jump_velocity_command_name": _JUMP_VELOCITY_COMMAND_NAME,
                "gait_velocity_command_name": _GAIT_VELOCITY_COMMAND_NAME,
                "num_groups": 2,
            },
        )

        # ===================== jump_commands obs group =====================
        # Feed to the jump teacher JIT together with policy + commands.
        # Non-jump group envs are zeroed out.
        self.observations.jump_commands = ObservarionsCfg.ExpertTargetCfg()
        self.observations.jump_commands.jump_command = ObsTerm(
            func=mdp.expert_separated_jump_commands,
            params={"jump_command_name": _JUMP_COMMAND_NAME, "num_groups": 2},
        )

        # ===================== gait_commands obs group =====================
        # Feed to the gait teacher JIT together with policy + commands.
        # Non-gait group envs are zeroed out.
        self.observations.gait_commands = ObservarionsCfg.ExpertTargetCfg()
        self.observations.gait_commands.gait_with_height = ObsTerm(
            func=mdp.expert_separated_gait_commands,
            params={
                "gait_command_name": "gait_command",
                "height_command_name": _JUMP_COMMAND_NAME,
                "height_index": _BASE_JUMP_STANDING_HEIGHT_INDEX,
                "num_groups": 2,
            },
        )

        # ===================== obsHistory_flat =====================
        # Flattened history for encoder-based teacher JITs.
        self.observations.obsHistory_flat = _HistoryObsFlatCfg()


@configclass
class WFMultiExpertFlatEnvCfg_PLAY(WFMultiExpertFlatEnvCfg):
    """Inference version: 32 envs, no randomisation, no curriculum."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.curriculum = None
        self.commands.base_velocity_active_debug.debug_vis = True

        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity_jump.ranges.lin_vel_x = (-2.0, 2.0)
        self.commands.base_velocity_jump.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity_gait.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity_gait.ranges.lin_vel_y = (-0.5, 0.5)
