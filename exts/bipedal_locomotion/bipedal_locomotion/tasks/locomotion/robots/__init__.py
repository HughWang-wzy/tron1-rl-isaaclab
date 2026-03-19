import gymnasium as gym

from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_ppo_cfg import PF_TRON1AFlatPPORunnerCfg, WF_TRON1AFlatPPORunnerCfg, SF_TRON1AFlatPPORunnerCfg, WF_TRON1APPORunnerCfg, WF_TRON1AJumpPPORunnerCfg, WF_TRON1AMoEPPORunnerCfg, WF_TRON1AGaitPPORunnerCfg

from . import limx_pointfoot_env_cfg, limx_wheelfoot_env_cfg, limx_solefoot_env_cfg, limx_wf_moe_env_cfg, limx_wheelfoot_gait_env_cfg, limx_wf_multiexpert_env_cfg

##
# Create PPO runners for RSL-RL
##

limx_pf_blind_flat_runner_cfg = PF_TRON1AFlatPPORunnerCfg()

limx_wf_blind_flat_runner_cfg = WF_TRON1AFlatPPORunnerCfg()

limx_sf_blind_flat_runner_cfg = SF_TRON1AFlatPPORunnerCfg()

limx_wf_runner_cfg = WF_TRON1APPORunnerCfg()

limx_wf_jump_runner_cfg = WF_TRON1AJumpPPORunnerCfg()

limx_wf_moe_runner_cfg = WF_TRON1AMoEPPORunnerCfg()

limx_wf_gait_runner_cfg = WF_TRON1AGaitPPORunnerCfg()


##
# Register Gym environments
##

############################
# PF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-PF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-PF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_pointfoot_env_cfg.PFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_pf_blind_flat_runner_cfg,
    },
)

#############################
# WF Blind Flat Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_blind_flat_runner_cfg,
    },
)


#############################
# WF Rough Environment
#############################
gym.register(
    id="Isaac-Limx-WF",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_runner_cfg,
    },
)

############################
# SF Blind Flat Environment
############################
gym.register(
    id="Isaac-Limx-SF-Blind-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-SF-Blind-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_solefoot_env_cfg.SFBlindFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_sf_blind_flat_runner_cfg,
    },
)

#############################
# WF Jump Flat Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Jump-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFJumpFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_jump_runner_cfg,
        "rsl_rl_distillation_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_distillation_cfg:WF_JumpDistillationCfg"
        ),
    },
)

gym.register(
    id="Isaac-Limx-WF-Jump-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFJumpFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_jump_runner_cfg,
        "rsl_rl_distillation_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_distillation_cfg:WF_JumpDistillationCfg"
        ),
    },
)

#############################
# WF Jump Rough Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Jump-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFJumpRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_jump_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Jump-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_env_cfg.WFJumpRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_jump_runner_cfg,
    },
)

#############################
# WF MoE Flat Environment
#############################
gym.register(
    id="Isaac-Limx-WF-MoE-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wf_moe_env_cfg.WFMoEFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_moe_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-MoE-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wf_moe_env_cfg.WFMoEFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_moe_runner_cfg,
    },
)

#############################
# WF Gait Flat Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Gait-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_gait_env_cfg.WFGaitFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_gait_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Gait-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_gait_env_cfg.WFGaitFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_gait_runner_cfg,
    },
)

#############################
# WF Gait Rough Environment
#############################
gym.register(
    id="Isaac-Limx-WF-Gait-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_gait_env_cfg.WFGaitRoughEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_gait_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Gait-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wheelfoot_gait_env_cfg.WFGaitRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_gait_runner_cfg,
    },
)

#############################
# WF MultiExpert Flat Environment  (Jump + Gait student for multi-expert distillation)
#############################
gym.register(
    id="Isaac-Limx-WF-MultiExpert-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wf_multiexpert_env_cfg.WFMultiExpertFlatEnvCfg,
        "rsl_rl_cfg_entry_point": limx_wf_jump_runner_cfg,  # placeholder; distillation uses its own runner
        "rsl_rl_distillation_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_distillation_cfg:WF_MultiExpertDistillationCfg"
        ),
    },
)

gym.register(
    id="Isaac-Limx-WF-MultiExpert-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": limx_wf_multiexpert_env_cfg.WFMultiExpertFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": limx_wf_jump_runner_cfg,  # placeholder; distillation uses its own runner
        "rsl_rl_distillation_cfg_entry_point": (
            "bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_distillation_cfg:WF_MultiExpertDistillationCfg"
        ),
    },
)
