import math

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.sensors import RayCasterCfg, patterns

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFBaseEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.WF.limx_base_env_cfg import RewardsCfg
from bipedal_locomotion.tasks.locomotion.cfg.WF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
)


############################
# Wheelfoot Gait Rewards
############################


@configclass
class WFGaitRewardsCfg(RewardsCfg):
    """Rewards for pure gait locomotion on the wheelfoot robot (no wheel usage)."""
    jump_height: RewTerm | None = None
    jump_landing: RewTerm | None = None
    jump_upward_vel: RewTerm | None = None
    jump_flight_vel: RewTerm | None = None
    jump_tuck: RewTerm | None = None
    keep_balance = None
    track_base_height: RewTerm | None = None
    pen_base_contact: RewTerm | None = None
    pen_feet_distance = None
    rew_leg_symmetry = None
    rew_same_foot_x_position = None
    pen_base_height: RewTerm | None = None
    
    
    # ---- survival ----
    termination = RewTerm(func=mdp.is_terminated, weight=-1000.0)
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-4,
        params={
            "target_joint_positions": {"abad_L_Joint": 0.0, "abad_R_Joint": 0.0},
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["wheel_L_Link", "wheel_R_Link"]),
        },
    )

    # ---- velocity tracking ----
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": 0.4},
    )
    rew_lin_vel_xy_enhanced = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    rew_ang_vel_z_enhanced = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    # ---- gait reward (force + velocity tracking for alternating contact) ----
    gait_reward = RewTerm(
        func=mdp.GaitReward,
        weight=2,
        params={
            "tracking_contacts_shaped_force": -2.0,
            "tracking_contacts_shaped_vel": -0.0,
            "gait_force_sigma": 1.0,
            "gait_vel_sigma": 0.25,
            "kappa_gait_probs": 0.05,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"),
            "swing_height_scale": 0.0,
            "foot_radius": 0.13,
            "max_swing_height": 0.3,
        },
    )
    
    gait_symmetry = RewTerm(
        func=mdp.GaitSymmetryReward,
        weight=-1.5,
        params={
            "command_name": "gait_command",
            "asset_cfg": SceneEntityCfg("robot", body_names=["wheel_L_Link", "wheel_R_Link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["wheel_L_Link", "wheel_R_Link"]),
            "contact_force_threshold": 1.0,
            "contact_weight": 1.0,
        },
    )
    # gait_symmetry = None

    # ---- standard penalties ----
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    pen_flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-8.0)
    track_base_height = RewTerm(
        func=mdp.track_base_height_from_command, weight=4.0,
        params={"command_name": "height_command", "sigma": 0.6},
    )
    track_base_height_enhanced = RewTerm(
        func=mdp.track_base_height_from_command, weight=4.0,
        params={"command_name": "height_command", "sigma": 0.2},
    )
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.01)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.000008)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    pen_joint_power_l1 = RewTerm(func=mdp.joint_powers_l1, weight=-5e-6)
    pen_non_wheel_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )

    # ---- heavily penalise wheel velocity (force gait, not rolling) ----
    pen_joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+")},
    )
    pen_vel_non_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )

    # ---- contact penalties ----
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 10.0,
        },
    )
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-1,
        params={"min_feet_distance": 0.2,
                "max_feet_distance": 0.4,
                "feet_links_name": ["wheel_[RL]_Link"]}
    )


############################
# Wheelfoot Gait Curriculum
############################


@configclass
class WFGaitCurriculumCfg:
    """Curriculum for gait environment."""

    lin_vel_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "rew_lin_vel_xy"},
    )
    ang_vel_levels = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={"reward_term_name": "rew_ang_vel_z"},
    )

    # wheel_vel_penalty_weight = CurrTerm(
    #     func=mdp.reward_weight_abs_curriculum,
    #     params={
    #         "term_name": "pen_joint_vel_wheel_l2",
    #         "reward_term_name": "pen_joint_vel_wheel_l2",
    #         "reward_threshold_ratio": 0.1,
    #         "step": 0.1,
    #         "max_abs_weight": 0.5,
    #         "min_interval_iterations": 500,
    #     },
    # )

############################
# Wheelfoot Gait Environment
############################


@configclass
class WFGaitFlatEnvCfg(WFBaseEnvCfg):
    """Wheelfoot robot using pure gait (no wheel rolling) on flat terrain.

    The robot walks using alternating leg stance like a biped.
    Wheel joints are heavily penalised to prevent rolling.
    No height map — blind flat environment.
    """

    def __post_init__(self):
        super().__post_init__()

        # ===================== no height scanner (blind) =====================
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # ===================== rewards =====================
        self.rewards = WFGaitRewardsCfg()

        self.terminations.low_base_height = DoneTerm(
            func=mdp.base_height_below_minimum,
            params={"minimum_height": 0.4},
        )
        
        # ===================== curriculum =====================
        self.curriculum = WFGaitCurriculumCfg()

        # ===================== gait command =====================
        self.commands.gait_command = mdp.UniformGaitCommandCfg(
            resampling_time_range=(5.0, 5.0),
            debug_vis=False,
            ranges=mdp.UniformGaitCommandCfg.Ranges(
                frequencies=(1.5, 2.5),
                offsets=(0.5, 0.5),
                durations=(0.5, 0.5),
                swing_height=(0.3, 0.3),
            ),
        )

        # ===================== height command =====================
        self.commands.height_command = mdp.HeightCommandCfg(
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
            min_height=0.70,
            max_height=0.90,
        )

        # ===================== observations =====================
        self.observations.commands.gait_command = ObsTerm(
            func=mdp.get_gait_command, params={"command_name": "gait_command"}
        )
        self.observations.commands.height_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "height_command"}
        )
        # ===================== velocity command =====================
        self.commands.base_velocity = mdp.UniformLevelVelocityCommandCfg(
            asset_name="robot",
            heading_command=True,
            heading_control_stiffness=1.0,
            rel_standing_envs=0.1,
            rel_heading_envs=1.0,
            debug_vis=True,
            resampling_time_range=(10.0, 10.0),
            ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-0.5, 0.5),
                lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-0.5, 0.5),
                heading=(-math.pi, math.pi),
            ),
            limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-math.pi/2, math.pi/2),
                heading=(-math.pi, math.pi),
            ),
        )

@configclass
class WFGaitFlatEnvCfg_PLAY(WFGaitFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 128
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.curriculum = None


############################
# Wheelfoot Gait Rough Environment
############################


@configclass
class WFGaitRoughEnvCfg(WFGaitFlatEnvCfg):
    """Wheelfoot robot using pure gait on rough terrain (waves, boxes, random_rough — no stairs).

    Height scanner enabled so the policy can perceive terrain.
    """

    def __post_init__(self):
        super().__post_init__()

        # ===================== height scanner =====================
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # ===================== rough terrain (no stairs) =====================
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class WFGaitRoughEnvCfg_PLAY(WFGaitRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.5, 0.5)
        self.curriculum = None

        # spawn randomly across terrain grid
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG
