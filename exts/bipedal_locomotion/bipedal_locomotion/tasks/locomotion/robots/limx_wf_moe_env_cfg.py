import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_cfg import WHEELFOOT_CFG
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFBaseEnvCfg, WFJumpCurriculumCfg
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg


############################
# Wheelfoot MoE Flat Environment
############################


@configclass
class WFMoEFlatEnvCfg(WFBaseEnvCfg):
    """WF MoE flat environment — combines wheeled and legged locomotion.

    Two MoE experts are expected to specialise into:
      - Expert 0: wheeled (rolling) locomotion — velocity tracking
      - Expert 1: legged (jumping) locomotion — explosive leg extension
    """

    def __post_init__(self):
        super().__post_init__()

        # -- blind flat: no height scanner, no terrain curriculum
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # -- replace curriculum with jump-aware version (disables terrain_levels)
        self.curriculum = WFJumpCurriculumCfg()
        self.curriculum.terrain_levels = None

        # -- increase action scale so legs can produce explosive jumps
        self.actions.joint_pos.scale = 0.5

        # ===================== jump commands =====================
        self.commands.base_jump = mdp.JumpCommandCfg(
            jump_probability=0.3,
            standing_height_range=(0.6, 0.9),
            jump_delta_range=(0.25, 0.5),
            jump_margin=0.5,
            resampling_time_range=(3.0, 10.0),
        )

        # ===================== observations =====================
        # jump command → policy needs to know when to jump
        self.observations.commands.jump_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_jump"}
        )

        # feet contact state → privileged critic observation
        self.observations.critic.feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")},
        )

        # ===================== rewards =====================
        # -- conditional penalties (suppressed during jumps)
        self.rewards.pen_lin_vel_z = RewTerm(
            func=mdp.conditional_lin_vel_z_l2,
            weight=-0.3,
            params={"command_name": "base_jump"},
        )
        self.rewards.pen_flat_orientation_l2 = RewTerm(
            func=mdp.conditional_flat_orientation_l2,
            weight=-10.0,
            params={"command_name": "base_jump", "jump_scale": 0.2},
        )
        self.rewards.pen_base_height = None

        self.rewards.track_base_height = RewTerm(
            func=mdp.track_base_height_exp,
            weight=2.0,
            params={"command_name": "base_jump", "sigma": 0.15},
        )

        # -- jump rewards
        self.rewards.jump_height = RewTerm(
            func=mdp.jump_height_reward,
            weight=20.0,
            params={
                "command_name": "base_jump",
                "sigma": 0.2,
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
            },
        )
        self.rewards.jump_upward_vel = RewTerm(
            func=mdp.jump_upward_vel,
            weight=10.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
            },
        )
        self.rewards.jump_tuck = RewTerm(
            func=mdp.jump_tuck_legs,
            weight=5.0,
            params={
                "command_name": "base_jump",
                "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
                "target_distance": 0.5,
                "sigma": 0.2,
            },
        )

        # -- tuned penalties for dual-mode operation
        self.rewards.pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
        self.rewards.pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.005)
        self.rewards.pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00002)
        self.rewards.pen_joint_vel_wheel_l2 = RewTerm(
            func=mdp.joint_vel_l2, weight=-1e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+")}
        )
        self.rewards.pen_vel_non_wheel_l2 = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-5e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
        )

        # -- joint pos limits: suppressed during jumps (robot needs full ROM)
        self.rewards.pen_non_wheel_pos_limits = RewTerm(
            func=mdp.conditional_joint_pos_limits,
            weight=-2.0,
            params={
                "command_name": "base_jump",
                "asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*"),
                "jump_scale": 0.0,
            },
        )

        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*"]),
                "threshold": 10.0,
            },
        )

        # -- base contact soft penalty
        self.rewards.pen_base_contact = RewTerm(
            func=mdp.base_contact_penalty,
            weight=-0.01,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
                "threshold": 10.0,
            },
        )

        # ===================== terminations =====================
        # only terminate when BOTH contact AND bad orientation (allow base contact during jumps)
        self.terminations.base_contact = DoneTerm(
            func=mdp.base_contact_and_bad_orientation,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
                "limit_angle": 1,
                "threshold": 1.0,
            },
        )

        # ===================== curriculum =====================
        self.curriculum.jump_probability = CurrTerm(
            func=mdp.jump_probability_curriculum,
            params={
                "command_name": "base_jump",
                "start_prob": 0.05,
                "end_prob": 0.5,
                "start_iteration": 500,
                "end_iteration": 5000,
                "num_steps_per_env": 24,
            },
        )

        # -- velocity command with limit_ranges for curriculum
        self.commands.base_velocity = mdp.UniformLevelVelocityCommandCfg(
            asset_name="robot",
            heading_command=True,
            heading_control_stiffness=1.0,
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            debug_vis=True,
            resampling_time_range=(10.0, 10.0),
            ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-1, 1),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
            limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-4.0, 4.0),
                lin_vel_y=(-0.0, 0.0),
                ang_vel_z=(-math.pi, math.pi),
                heading=(-math.pi, math.pi),
            ),
        )


@configclass
class WFMoEFlatEnvCfg_PLAY(WFMoEFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None

        # higher jump probability for demo
        self.commands.base_jump.jump_probability = 0.8
        self.commands.base_jump.resampling_time_range = (5.0, 5.0)
        # no assist force during play
        self.commands.base_jump.assist_force_max = 0.0
        self.curriculum = None
