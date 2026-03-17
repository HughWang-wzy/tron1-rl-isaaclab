from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from bipedal_locomotion.utils.wrappers.rsl_rl.rl_mlp_cfg import EncoderCfg, RslRlPpoAlgorithmMlpCfg, MoEActorCriticCfg, MoEPpoAlgorithmCfg

import os
robot_type = os.getenv("ROBOT_TYPE")

# Isaac Lab original RSL-RL configuration
@configclass
class PFPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "pf_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
#-----------------------------------------------------------------
@configclass
class PF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 200
    experiment_name = "pf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )

#-----------------------------------------------------------------
@configclass
class SF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 15000
    save_interval = 500
    experiment_name = "sf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON1AFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "wf_tron_1a_flat"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON1AJumpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "wf_tron_1a_jump"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON1APPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "wf_tron_1a"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach = True,
        num_output_dim = 3,
        hidden_dims = [256, 128],
        activation = "elu",
        orthogonal_init = False,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON1AMoEPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """WF MoE PPO runner — uses MoEOnPolicyRunner."""

    class_name: str = "MoEOnPolicyRunner"
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 500
    experiment_name = "wf_tron_1a_moe"
    empirical_normalization = False
    policy = MoEActorCriticCfg(
        num_experts=2,
        top_k=1,
        gating_hidden_dims=[128, 64],
        expert_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    algorithm = MoEPpoAlgorithmCfg(
        class_name="MoEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
        moe_aux_loss_coef=0.1,
    )
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=3,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )


#-----------------------------------------------------------------
@configclass
class WF_TRON1AGaitPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "wf_tron_1a_gait"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmMlpCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        obs_history_len=10,
    )
    encoder = EncoderCfg(
        output_detach=True,
        num_output_dim=3,
        hidden_dims=[256, 128],
        activation="elu",
        orthogonal_init=False,
    )
