"""Multi-expert distillation student environment: Jump + Gait.

``WFMultiExpertFlatEnvCfg`` merges the Jump and Gait environments into a
single student environment for multi-expert distillation training.

Environment split (static, by index):
  envs  0 .. N//2 - 1  →  group 0  (Jump expert)
  envs  N//2 .. N - 1  →  group 1  (Gait expert)

Observation groups
------------------
policy
    Student proprioceptive input (concatenated in policy group).
env_group
    Per-env expert id. Shape (N, 1), float. 0 = jump, 1 = gait.
    Consumed by MultiExpertDistillation via ``teacher_id_obs_group``.
critic
    Combined privileged obs for both teachers (contains jump + gait specific
    fields so either teacher can find its slice).
commands
    Group-conditioned command group with separated sampling:
    jump envs use jump velocity + jump command, gait envs use gait velocity.
jump_commands
    Jump command only — feed to jump teacher JIT.
gait_commands
    Gait command + standing_height(from base_jump) — feed to gait teacher JIT.
obsHistory_flat
    Flattened proprioceptive history, shape (N, 10 * policy_obs_dim).
    Use as encoder input when teacher JIT wraps encoder + actor.

Configuring teacher JIT obs_groups (in WF_MultiExpertDistillationCfg)
----------------------------------------------------------------------
If teachers were exported as  encoder + actor  combined JITs:
    jump_expert.obs_groups = ["obsHistory_flat", "policy", "commands"]
    gait_expert.obs_groups = ["obsHistory_flat", "policy", "commands", "gait_commands"]

If teachers are actor-only JITs exported from on-policy training:
    jump_expert.obs_groups = ["policy", "commands", "jump_commands"]
    gait_expert.obs_groups = ["policy", "commands", "gait_commands"]
"""

import math

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFJumpFlatEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.WF.limx_base_env_cfg import ObservarionsCfg

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
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 10
        self.flatten_history_dim = True


# ---------------------------------------------------------------------------
# Combined multi-expert student environment
# ---------------------------------------------------------------------------

@configclass
class WFMultiExpertFlatEnvCfg(WFJumpFlatEnvCfg):
    """Student environment for Jump + Gait multi-expert distillation.

    Inherits the Jump setup (jump commands, jump rewards, action_scale=0.5,
    no height scanner) and adds:
      - Separated jump/gait velocity command samplers.
      - Gait command for group-1 envs.
      - Gait height condition reusing ``base_jump[:, 2]`` (standing_height).
      - ``env_group`` obs (static 0/1 split by env index).
      - ``jump_commands`` / ``gait_commands`` separate obs groups.
      - ``obsHistory_flat`` for encoder-based teacher JITs.
    """

    def __post_init__(self):
        super().__post_init__()   # full Jump flat env setup

        # Disable all curriculum terms for multi-expert distillation runs.
        self.curriculum = None
        self.rewards = None
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
                lin_vel_x=(-1.0, 1.0),
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
        self.commands.base_velocity_gait = mdp.UniformLevelVelocityCommandCfg(
            asset_name="robot",
            heading_command=True,
            heading_control_stiffness=1.0,
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            debug_vis=False,
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
        # Keep teacher ``commands`` input dim=6 while sampling jump/gait velocity
        # commands from separate command terms by env-group.
        self.observations.commands = ObservarionsCfg.ExpertTargetCfg()
        self.observations.commands.expert_commands = ObsTerm(
            func=mdp.expert_separated_commands,
            params={
                "jump_velocity_command_name": _JUMP_VELOCITY_COMMAND_NAME,
                "gait_velocity_command_name": _GAIT_VELOCITY_COMMAND_NAME,
                "jump_command_name": _JUMP_COMMAND_NAME,
                "num_groups": 2,
            },
        )

        # ===================== jump_commands obs group =====================
        # Feed to the jump teacher JIT together with policy + commands.
        self.observations.jump_commands = ObservarionsCfg.ExpertTargetCfg()
        self.observations.jump_commands.jump_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_jump"},
        )

        # ===================== gait_commands obs group =====================
        # Feed to the gait teacher JIT together with policy + commands.
        self.observations.gait_commands = ObservarionsCfg.ExpertTargetCfg()
        self.observations.gait_commands.gait_command = ObsTerm(
            func=mdp.get_gait_command,
            params={"command_name": "gait_command"},
        )
        # Keep gait height conditioned on jump command's standing_height so
        # both experts share one height source in the merged environment.
        self.observations.gait_commands.height_command = ObsTerm(
            func=mdp.command_component,
            params={"command_name": "base_jump", "index": _BASE_JUMP_STANDING_HEIGHT_INDEX},
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
