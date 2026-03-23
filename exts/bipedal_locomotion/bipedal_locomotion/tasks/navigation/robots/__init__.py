import gymnasium as gym

from bipedal_locomotion.tasks.navigation.agents.rsl_rl_ppo_cfg import WFNavigationPPORunnerCfg
from . import wf_navigation_env_cfg

##
# Create PPO runner
##

wf_nav_runner_cfg = WFNavigationPPORunnerCfg()

##
# Register Gym environments
##

##############################
# WF Navigation with Obstacles
##############################

gym.register(
    id="Isaac-Limx-WF-Nav-Obstacles-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wf_navigation_env_cfg.WFNavObstaclesEnvCfg,
        "rsl_rl_cfg_entry_point": wf_nav_runner_cfg,
    },
)

gym.register(
    id="Isaac-Limx-WF-Nav-Obstacles-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": wf_navigation_env_cfg.WFNavObstaclesEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": wf_nav_runner_cfg,
    },
)
