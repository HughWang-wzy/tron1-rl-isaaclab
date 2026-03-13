import math

from isaaclab.utils import configclass

from bipedal_locomotion.assets.config.wheelfoot_cfg import WHEELFOOT_CFG
from bipedal_locomotion.tasks.locomotion.robots.limx_wheelfoot_env_cfg import WFBaseEnvCfg, WFBaseEnvCfg_PLAY


############################
# Wheelfoot MoE Flat Environment
############################


@configclass
class WFMoEFlatEnvCfg(WFBaseEnvCfg):
    """WF MoE flat environment for validating MoE architecture."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None
        self.curriculum.terrain_levels = None


@configclass
class WFMoEFlatEnvCfg_PLAY(WFBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        self.scene.height_scanner = None
        self.observations.policy.heights = None
        self.observations.critic.heights = None
        self.curriculum.terrain_levels = None
