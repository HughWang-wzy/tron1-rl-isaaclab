import math

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
)

from bipedal_locomotion.assets.config.wheelfoot_cfg import WHEELFOOT_CFG
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFBaseEnvCfg, WFJumpCurriculumCfg
from bipedal_locomotion.tasks.locomotion.cfg.WF.limx_base_env_cfg import RewardsCfg, ObservarionsCfg
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise


############################
# MoE Mixed Terrain Config
############################

MOE_MIXED_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

MOE_MIXED_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)


############################
# Wheelfoot MoE Rewards
############################


@configclass
class WFMoERewardsCfg(RewardsCfg):
    """MoE reward config — inherits RewardsCfg for type compatibility, overrides all fields."""

    # ---- survival ----
    keep_balance = RewTerm(func=mdp.stay_alive, weight=1.0)
    stand_still = RewTerm(func=mdp.stand_still, weight=-5.0)

    # ---- velocity tracking ----
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.2)},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ---- posture ----
    rew_leg_symmetry = RewTerm(
        func=mdp.leg_symmetry, weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"), "std": math.sqrt(0.5)},
    )
    rew_same_foot_x_position = RewTerm(
        func=mdp.same_feet_x_position, weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")},
    )

    # ---- height tracking (jump-aware) ----
    track_base_height = RewTerm(
        func=mdp.track_base_height_exp, weight=2.0,
        params={"command_name": "base_jump", "sigma": 0.15},
    )

    # ---- jump rewards ----
    jump_height = RewTerm(
        func=mdp.jump_height_reward, weight=20.0,
        params={
            "command_name": "base_jump",
            "sigma": 0.2,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
        },
    )
    jump_upward_vel = RewTerm(
        func=mdp.jump_upward_vel, weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
        },
    )
    jump_flight_vel = RewTerm(
        func=mdp.jump_flight_vel_tracking, weight=2.0,
        params={
            "command_name": "base_jump",
            "velocity_command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
            "std": 0.25,
        },
    )
    jump_tuck = RewTerm(
        func=mdp.jump_tuck_legs, weight=5.0,
        params={
            "command_name": "base_jump",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
            "tuck_angles": {
                "hip_L_Joint": 1.15,
                "knee_L_Joint": 1.15,
                "hip_R_Joint": -1.15,
                "knee_R_Joint": -1.15,
            },
            "sigma": 0.25,
        },
    )

    # ---- conditional penalties (suppressed during jumps) ----
    pen_lin_vel_z = RewTerm(
        func=mdp.conditional_lin_vel_z_l2, weight=-0.3,
        params={"command_name": "base_jump"},
    )
    pen_flat_orientation_l2 = RewTerm(
        func=mdp.conditional_flat_orientation_l2, weight=-10.0,
        params={"command_name": "base_jump", "jump_scale": 0.2},
    )
    pen_non_wheel_pos_limits = RewTerm(
        func=mdp.conditional_joint_pos_limits, weight=-2.0,
        params={
            "command_name": "base_jump",
            "asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*"),
            "jump_scale": 0.0,
        },
    )

    # ---- standard penalties ----
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.005)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00002)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-7)
    pen_joint_power_l1 = RewTerm(func=mdp.joint_powers_l1, weight=-2e-5)
    pen_joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+")},
    )
    pen_vel_non_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance, weight=-100,
        params={"min_feet_distance": 0.32, "max_feet_distance": 0.35, "feet_links_name": ["wheel_[RL]_Link"]},
    )

    # ---- contact penalties ----
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*"]),
            "threshold": 10.0,
        },
    )
    pen_base_contact = RewTerm(
        func=mdp.base_contact_penalty, weight=-0.01,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
            "threshold": 10.0,
        },
    )

    # ---- terrain-adaptive rewards (uphill stairs or lateral velocity) ----
    pen_terrain_wheel_vel = RewTerm(
        func=mdp.terrain_adaptive_wheel_penalty, weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+"),
            "command_name": "base_velocity",
            "gradient_threshold": 0.02,
            "step_edge_threshold": 0.03,
            "lateral_vel_threshold": 0.1,
        },
    )
    rew_terrain_gait = RewTerm(
        func=mdp.terrain_adaptive_gait_reward, weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
            "command_name": "base_velocity",
            "gradient_threshold": 0.02,
            "step_edge_threshold": 0.03,
            "lateral_vel_threshold": 0.1,
            "air_time_threshold": 0.5,
        },
    )

    # ---- explicitly disabled (overridden from base) ----
    pen_base_height: RewTerm | None = None
    jump_landing: RewTerm | None = None


############################
# Wheelfoot MoE Environment
############################


@configclass
class WFMoEFlatEnvCfg(WFBaseEnvCfg):
    """WF MoE environment — combines wheeled and legged locomotion on mixed terrain.

    Two MoE experts are expected to specialise into:
      - Expert 0: wheeled (rolling) locomotion — velocity tracking on flat ground
      - Expert 1: legged (jumping) locomotion — stair climbing / explosive leg extension
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

        # ===================== height observations =====================
        self.observations.policy.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=GaussianNoise(mean=0.0, std=0.01),
            clip=(-100.0, 100.0),
        )
        self.observations.critic.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-100.0, 100.0),
        )
        self.observations.obsHistory.heights = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-100.0, 100.0),
        )

        # ===================== mixed terrain =====================
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = MOE_MIXED_TERRAINS_CFG

        # ===================== rewards =====================
        self.rewards = WFMoERewardsCfg()

        # ===================== curriculum =====================
        self.curriculum = WFJumpCurriculumCfg()
        # keep terrain_levels for mixed terrain curriculum
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

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
        # gait phase clock → reference signal for alternating leg swing
        self.observations.commands.gait_phase = ObsTerm(
            func=mdp.gait_phase, params={"period": 0.7}
        )

        # feet contact state → privileged critic observation
        self.observations.critic.feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")},
        )

        # ===================== expert target (router supervision) =====================
        self.observations.expert_target = ObservarionsCfg.ExpertTargetCfg()
        self.observations.expert_target.gait_score = ObsTerm(
            func=mdp.gait_needed_score,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "command_name": "base_velocity",
                "gradient_threshold": 0.02,
                "step_edge_threshold": 0.03,
                "lateral_vel_threshold": 0.1,
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

        # mixed terrain play
        self.scene.terrain.terrain_generator = MOE_MIXED_TERRAINS_PLAY_CFG
        self.scene.terrain.max_init_terrain_level = None

        # higher jump probability for demo
        self.commands.base_velocity.ranges.lin_vel_x = (-3.0, 3.0)
        self.commands.base_jump.jump_probability = 1
        self.commands.base_jump.resampling_time_range = (3.0, 3.0)
        # no assist force during play
        self.commands.base_jump.assist_force_max = 0.0
        self.curriculum = None
