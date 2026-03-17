"""Multi-expert distillation student environment: Jump + Gait.

``WFMultiExpertFlatEnvCfg`` merges the Jump and Gait environments into a
single student environment for multi-expert distillation training.

Environment split (static, by index):
  envs  0 .. N//2 - 1  →  group 0  (Jump expert)
  envs  N//2 .. N - 1  →  group 1  (Gait expert)

Observation groups
------------------
policy
    Student input (~22 dims, no privileged info).
env_group
    Per-env expert id. Shape (N, 1), float. 0 = jump, 1 = gait.
    Consumed by MultiExpertDistillation via ``teacher_id_obs_group``.
critic
    Combined privileged obs for both teachers (contains jump + gait specific
    fields so either teacher can find its slice).
commands
    Shared base velocity command.
jump_commands
    Jump command only — feed to jump teacher JIT.
gait_commands
    Gait command + height command — feed to gait teacher JIT.
obsHistory_flat
    Flattened proprioceptive history, shape (N, 10 * policy_obs_dim).
    Use as encoder input when teacher JIT wraps encoder + actor.

Configuring teacher JIT obs_groups (in WF_MultiExpertDistillationCfg)
----------------------------------------------------------------------
If teachers were exported as  encoder + actor  combined JITs:
    jump_expert.obs_groups = ["obsHistory_flat", "policy", "commands", "jump_commands"]
    gait_expert.obs_groups = ["obsHistory_flat", "policy", "commands", "gait_commands"]

If teachers are plain MLPs taking policy + privileged obs:
    jump_expert.obs_groups = ["policy", "critic"]
    gait_expert.obs_groups = ["policy", "critic"]
"""

import math

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from bipedal_locomotion.tasks.locomotion import mdp
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFJumpFlatEnvCfg
from bipedal_locomotion.tasks.locomotion.cfg.WF.limx_base_env_cfg import ObservarionsCfg


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
      - Gait commands (gait_command, height_command) for group-1 envs.
      - ``env_group`` obs (static 0/1 split by env index).
      - ``jump_commands`` / ``gait_commands`` separate obs groups.
      - ``obsHistory_flat`` for encoder-based teacher JITs.
    """

    def __post_init__(self):
        super().__post_init__()   # full Jump flat env setup

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
        self.commands.height_command = mdp.HeightCommandCfg(
            resampling_time_range=(10.0, 10.0),
            debug_vis=False,
            min_height=0.70,
            max_height=0.90,
        )

        # ===================== env_group obs =====================
        # Shape (N, 1).  0.0 = jump expert,  1.0 = gait expert.
        self.observations.env_group = ObservarionsCfg.ExpertTargetCfg()
        self.observations.env_group.group_id = ObsTerm(
            func=mdp.env_group_id,
            params={"num_groups": 2},
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
        self.observations.gait_commands.height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "height_command"},
        )

        # ===================== extra critic obs (for gait teacher) =====================
        # Gait-teacher needs gait_command + height_command in privileged obs.
        self.observations.critic.gait_command = ObsTerm(
            func=mdp.get_gait_command,
            params={"command_name": "gait_command"},
        )
        self.observations.critic.height_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "height_command"},
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
