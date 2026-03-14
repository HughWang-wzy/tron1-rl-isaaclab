import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_cfg import WHEELFOOT_CFG
from bipedal_locomotion.tasks.locomotion.cfg.WF.limx_base_env_cfg import WFEnvCfg, CurriculumCfg
from bipedal_locomotion.tasks.locomotion.cfg.WF.terrains_cfg import (
    BLIND_ROUGH_TERRAINS_CFG,
    BLIND_ROUGH_TERRAINS_PLAY_CFG,
    STAIRS_TERRAINS_CFG,
    STAIRS_TERRAINS_PLAY_CFG,
)

from isaaclab.sensors import RayCasterCfg, patterns
from bipedal_locomotion.tasks.locomotion import mdp
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as GaussianNoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg


######################
# Wheelfoot Base Environment
######################


@configclass
class WFBaseEnvCfg(WFEnvCfg):
    def __post_init__(self):
        super().__post_init__()

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
        
        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class WFBaseEnvCfg_PLAY(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 32

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None
        # remove random base mass addition event
        self.events.add_base_mass = None


############################
# Wheelfoot Blind Flat Environment
############################


@configclass
class WFBlindFlatEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


@configclass
class WFBlindFlatEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.curriculum.terrain_levels = None


#############################
# Wheelfoot Blind Rough Environment
#############################


@configclass
class WFBlindRoughEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG


@configclass
class WFBlindRoughEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None
        
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG
        

##############################
# Wheelfoot Blind Stairs Environment
##############################

@configclass
class WFBlindStairEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class WFBlindStairEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))
        
        
#############################
# Wheelfoot Flat Environment
#############################

@configclass
class WFFlatEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.curriculum.terrain_levels = None

@configclass
class WFFlatEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.curriculum.terrain_levels = None
        
        
#############################
# Wheelfoot Rough Environment
#############################

@configclass
class WFRoughEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),clip=(-100.0, 100.0)
        )
        
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},clip=(-100.0, 100.0)
        )
        
        self.observations.obsHistory.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},clip=(-100.0, 100.0)
        )
        
        self.observations.commands.height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
        self.commands.base_height = mdp.HeightCommandCfg(min_height=0.6,max_height=0.9,resampling_time_range=(3.0, 15.0),)
        
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_CFG

        # update viewport camera
        self.viewer.origin_type = "env"


@configclass
class WFRoughEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = BLIND_ROUGH_TERRAINS_PLAY_CFG



        
        
##############################
# Wheelfoot Blind Stairs Environment
##############################


@configclass
class WFStairEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
                    noise=GaussianNoise(mean=0.0, std=0.01),
                    clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 6, math.pi / 6)

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class WFStairEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_Link",
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.5, 0.5]), #TODO: adjust size to fit real robot
            debug_vis=True,
            mesh_prim_paths=["/World/ground"],
        )
        self.observations.policy.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.observations.critic.heights = ObsTerm(func=mdp.height_scan,
            params = {"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip = (0.0, 10.0),
        )
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)

        self.events.reset_robot_base.params["pose_range"]["yaw"] = (-0.0, 0.0)

        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_PLAY_CFG.replace(difficulty_range=(0.5, 0.5))


#############################
# Wheelfoot Jump Flat Environment
#############################


@configclass
class WFJumpCurriculumCfg(CurriculumCfg):
    """Curriculum config for the jump environment — extends base with jump-specific terms."""

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
    # disable_base_contact_termination = CurrTerm(
    #     func=mdp.disable_termination,
    #     params={
    #         "term_name": "base_contact",
    #         "num_steps": 1500 * 24,
    #     },
    # )

@configclass
class WFJumpFlatEnvCfg(WFBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # no height scanner for blind jump
        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None

        # replace curriculum with jump-specific version (disables terrain_levels)
        self.curriculum = WFJumpCurriculumCfg()
        self.curriculum.terrain_levels = None

        # increase action scale for explosive leg extension
        self.actions.joint_pos.scale = 0.5

        # -- jump command
        self.commands.base_jump = mdp.JumpCommandCfg(
            jump_probability=0.3,
            standing_height_range=(0.6, 0.9),
            jump_delta_range=(0.25, 0.5),
            jump_margin=0.5,
            resampling_time_range=(3.0, 10.0),
        )

        # -- jump command observation (policy needs to know when to jump)
        self.observations.commands.jump_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_jump"}
        )

        # -- feet contact state as privileged observation (critic only)
        self.observations.critic.feet_contact = ObsTerm(
            func=mdp.feet_contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*")},
        )

        # -- replace conflicting rewards with conditional versions
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
            params={"command_name": "base_jump", "sigma": 0.1},
        )

        # -- jump rewards
        self.rewards.jump_height = RewTerm(
            func=mdp.jump_height_reward,
            weight=20.0,
            params={"command_name": "base_jump", "sigma": 0.1,"sensor_cfg": SceneEntityCfg("contact_forces", body_names="wheel_.*"),},
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
        self.rewards.pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
        self.rewards.pen_action_smoothness = RewTerm(func=mdp.ActionSmoothnessPenalty, weight=-0.005)
        # self.rewards.pen_feet_distance = None
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

        # -- base_contact soft penalty (threshold lowered to 10N to be meaningful)
        self.rewards.pen_base_contact = RewTerm(
            func=mdp.base_contact_penalty,
            weight=-0.01,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_Link"),
                "threshold": 10.0,
            },
        )

        # -- replace base_contact termination: only terminate when BOTH contact AND bad orientation
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
class WFJumpFlatEnvCfg_PLAY(WFJumpFlatEnvCfg):
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
