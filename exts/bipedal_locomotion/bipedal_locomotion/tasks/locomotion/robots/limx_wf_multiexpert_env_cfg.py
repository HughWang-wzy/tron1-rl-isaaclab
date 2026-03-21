"""Multi-expert distillation student environment: Jump + Gait.

"""

import math
from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg
from isaaclab.sim import DomeLightCfg, MdlFileCfg, RigidBodyMaterialCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise

from bipedal_locomotion.assets.config.wheelfoot_cfg import WHEELFOOT_CFG
from bipedal_locomotion.tasks.locomotion import mdp


##################
# Scene Definition
##################


@configclass
class WFSceneCfg(InteractiveSceneCfg):
    """Configuration for the test scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
            + "TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = MISSING
    height_scanner: RayCasterCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True, update_period=0.0
    )


##############
# MDP settings
##############


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=1.0,
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        debug_vis=True,
        resampling_time_range=(3.0, 15.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 0.7), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-math.pi, math.pi), heading=(-math.pi, math.pi)
        ),
    )
    base_jump: mdp.JumpCommandCfg | None = None
    height_command: mdp.HeightCommandCfg | None = None
    gait_command: mdp.UniformGaitCommandCfg | None = None


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["abad_L_Joint", "abad_R_Joint", "hip_L_Joint", "hip_R_Joint", "knee_L_Joint", "knee_R_Joint"],
        scale=0.25,
        use_default_offset=True,
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["wheel_L_Joint", "wheel_R_Joint"],
        scale=1.0,
    )


@configclass
class ObservarionsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_exclude_wheel,
            params={"wheel_joints_name": ["wheel_[RL]_Joint"]},
            noise=GaussianNoise(mean=0.0, std=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05
        )
        last_action = ObsTerm(
            func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HistoryObsCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=GaussianNoise(mean=0.0, std=0.05), clip=(-100.0, 100.0), scale=0.25
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=GaussianNoise(mean=0.0, std=0.025), clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_exclude_wheel,
            params={"wheel_joints_name": ["wheel_[RL]_Joint"]},
            noise=GaussianNoise(mean=0.0, std=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=0.05
        )
        last_action = ObsTerm(
            func=mdp.last_action, noise=GaussianNoise(mean=0.0, std=0.01), clip=(-100.0, 100.0), scale=1.0
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 10
            self.flatten_history_dim = False

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, clip=(-100.0, 100.0), scale=1.0)
        joint_vel = ObsTerm(func=mdp.joint_vel, clip=(-100.0, 100.0), scale=1.0)
        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        heights = ObsTerm(func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        robot_joint_torque = ObsTerm(func=mdp.robot_joint_torque)
        robot_joint_acc = ObsTerm(func=mdp.robot_joint_acc)
        feet_lin_vel = ObsTerm(func=mdp.feet_lin_vel, params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")})
        robot_mass = ObsTerm(func=mdp.robot_mass)
        robot_inertia = ObsTerm(func=mdp.robot_inertia)
        robot_joint_pos = ObsTerm(func=mdp.robot_joint_pos)
        robot_joint_stiffness = ObsTerm(func=mdp.robot_joint_stiffness)
        robot_joint_damping = ObsTerm(func=mdp.robot_joint_damping)
        robot_pos = ObsTerm(func=mdp.robot_pos)
        robot_vel = ObsTerm(func=mdp.robot_vel)
        robot_material_properties = ObsTerm(func=mdp.robot_material_properties)
        feet_contact_force = ObsTerm(
            func=mdp.robot_contact_force, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")}
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CommandsObsCfg(ObsGroup):
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

    @configclass
    class ExpertTargetCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    commands: CommandsObsCfg = CommandsObsCfg()
    obsHistory: HistoryObsCfg = HistoryObsCfg()
    expert_target: ExpertTargetCfg | None = None


@configclass
class EventsCfg:
    """Configuration for events."""

    prepare_quantity_for_tron1_piper = EventTerm(
        func=mdp.prepare_quantity_for_tron,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    radomize_rigid_body_mass_inertia = EventTerm(
        func=mdp.randomize_rigid_body_mass_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_inertia_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.2),
            "dynamic_friction_range": (0.7, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (32, 48),
            "damping_distribution_params": (2.0, 3.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_rigid_body_coms,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "com_distribution_params": ((-0.075, 0.075), (-0.075, 0.075), (-0.075, 0.075)),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.2, 0.2), "velocity_range": (-0.5, 0.5)},
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.5, 2.0),
            "damping_distribution_params": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            "force_range": {"x": (-500.0, 500.0), "y": (-500.0, 500.0), "z": (-0.0, 0.0)},
            "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
            "probability": 0.002,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    keep_balance = RewTerm(func=mdp.stay_alive, weight=1.0)
    stand_still = RewTerm(func=mdp.stand_still, weight=-5.0)
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=3.0, params={"command_name": "base_velocity", "std": math.sqrt(0.2)}
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    rew_leg_symmetry = RewTerm(
        func=mdp.leg_symmetry,
        weight=0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*"), "std": math.sqrt(0.5)},
    )
    rew_same_foot_x_position = RewTerm(
        func=mdp.same_feet_x_position,
        weight=-10,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")},
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.3)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.3)
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.00016)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-1.5e-7)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.3)
    pen_non_wheel_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["abad_.*", "hip_.*", "knee_.*", "base_Link"]),
            "threshold": 10.0,
        },
    )
    pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.03)
    pen_flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-12.0)
    pen_feet_distance = RewTerm(
        func=mdp.feet_distance,
        weight=-100,
        params={"min_feet_distance": 0.32, "max_feet_distance": 0.35, "feet_links_name": ["wheel_[RL]_Link"]},
    )
    pen_base_height = RewTerm(func=mdp.base_com_height, params={"target_height": 0.80}, weight=-30.0)
    pen_joint_power_l1 = RewTerm(func=mdp.joint_powers_l1, weight=-2e-5)
    pen_joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-5e-3, params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+")}
    )
    pen_vel_non_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )
    jump_height: RewTerm | None = None
    jump_landing: RewTerm | None = None
    jump_upward_vel: RewTerm | None = None
    jump_flight_vel: RewTerm | None = None
    jump_tuck: RewTerm | None = None
    track_base_height: RewTerm | None = None
    pen_base_contact: RewTerm | None = None


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

_BASE_JUMP_STANDING_HEIGHT_INDEX = 2
_JUMP_VELOCITY_COMMAND_NAME = "base_velocity_jump"
_GAIT_VELOCITY_COMMAND_NAME = "base_velocity_gait"
_JUMP_COMMAND_NAME = "base_jump"


# ---------------------------------------------------------------------------
# Local wheelfoot base/jump env
# ---------------------------------------------------------------------------


@configclass
class WFBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Local base wheelfoot env cfg for multi-expert file."""

    scene: WFSceneCfg = WFSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservarionsCfg = ObservarionsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = 2 * self.decimation
        self.sim.dt = 0.005
        self.seed = 42

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.scene.robot = WHEELFOOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {
            "abad_L_Joint": 0.0,
            "abad_R_Joint": 0.0,
            "hip_L_Joint": 0.0,
            "hip_R_Joint": 0.0,
            "knee_L_Joint": 0.0,
            "knee_R_Joint": 0.0,
        }

        self.events.add_base_mass.params["asset_cfg"].body_names = "base_Link"
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_Link"
        self.viewer.origin_type = "env"


@configclass
class WFJumpCurriculumCfg(CurriculumCfg):
    """Curriculum config for jump environment."""

    lin_vel_levels = CurrTerm(
        func=mdp.lin_vel_cmd_levels,
        params={"reward_term_name": "rew_lin_vel_xy"},
    )
    ang_vel_levels = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={"reward_term_name": "rew_ang_vel_z"},
    )
    jump_probability = CurrTerm(
        func=mdp.jump_probability_curriculum,
        params={
            "command_name": "base_jump",
            "start_prob": 0.05,
            "end_prob": 0.3,
            "start_iteration": 0,
            "end_iteration": 3000,
            "num_steps_per_env": 24,
        },
    )
    jump_assist_force = CurrTerm(
        func=mdp.jump_assist_force_curriculum,
        params={
            "command_name": "base_jump",
            "force_max": 300.0,
            "decay_start_iteration": 2000,
            "decay_per_1000_iter": 0.5,
            "num_steps_per_env": 24,
        },
    )


@configclass
class WFJumpFlatEnvCfg(WFBaseEnvCfg):
    """Local jump-flat env used as base for multi-expert student."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum = WFJumpCurriculumCfg()
        self.curriculum.terrain_levels = None
        self.actions.joint_pos.scale = 0.5

        self.commands.base_jump = mdp.JumpCommandCfg(
            jump_probability=0.3,
            standing_height_range=(0.7, 0.9),
            jump_delta_range=(0.25, 0.5),
            jump_margin=0.5,
            resampling_time_range=(3.0, 10.0),
        )

        self.observations.commands.jump_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_jump"}
        )
        self.observations.critic.feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")},
        )

        self.rewards.pen_lin_vel_z = RewTerm(
            func=mdp.conditional_lin_vel_z_l2, weight=-0.3, params={"command_name": "base_jump"}
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
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")},
        )
        self.rewards.jump_flight_vel = RewTerm(
            func=mdp.jump_flight_vel_tracking,
            weight=2.0,
            params={
                "command_name": "base_jump",
                "velocity_command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),
                "std": 0.25,
            },
        )
        self.rewards.jump_tuck = RewTerm(
            func=mdp.jump_tuck_legs,
            weight=5.0,
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
        self.rewards.pen_base_contact = RewTerm(
            func=mdp.base_contact_penalty,
            weight=-0.01,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"), "threshold": 10.0},
        )

        self.terminations.base_contact = DoneTerm(
            func=mdp.base_contact_and_bad_orientation,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
                "limit_angle": 1,
                "threshold": 1.0,
            },
        )
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
        self.events.push_robot = EventTerm(
            func=mdp.apply_external_force_torque_stochastic_additional,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
                "force_range": {"x": (-500.0, 500.0), "y": (-500.0, 500.0), "z": (-0.0, 0.0)},
                "torque_range": {"x": (-50.0, 50.0), "y": (-50.0, 50.0), "z": (-0.0, 0.0)},
                "probability": 0.002,
            },
            is_global_time=False,
            min_step_count_between_reset=0,
        )


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

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity_jump.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity_jump.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity_gait.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity_gait.ranges.lin_vel_y = (-0.5, 0.5)
