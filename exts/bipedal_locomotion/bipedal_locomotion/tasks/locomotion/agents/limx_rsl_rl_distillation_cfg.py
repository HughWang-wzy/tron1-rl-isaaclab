import os


WF_JumpDistillationCfg: dict = {
    "class_name": "DistillationRunner",
    "policy": {
        "class_name": "StudentTeacher",
        "student_hidden_dims": [512, 256, 128],
        "teacher_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "student_obs_normalization": False,
        "teacher_obs_normalization": False,
    },
    "algorithm": {
        "class_name": "Distillation",
        "num_learning_epochs": 1,
        "gradient_length": 15,
        "learning_rate": 1e-3,
        "loss_type": "mse",
        "max_grad_norm": 1.0,
    },
    "obs_groups": {
        "policy": ["policy"],
        "teacher": ["critic"],
    },
    "num_steps_per_env": 24,
    "save_interval": 200,
    "max_iterations": 5000,
    "experiment_name": "wf_jump_distillation",
}

_JUMP_JIT_PATH = os.path.expanduser(
    "./logs/rsl_rl/wf_tron_1a_jump/2026-04-06_00-06-47/exported/policy_all.pt"
)
_GAIT_JIT_PATH = os.path.expanduser(
    # "./logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/policy_all.pt"
    "./logs/rsl_rl/wf_tron_1a_gait/2026-03-24_03-38-42/exported/policy_all.pt"
)

WF_MultiExpertDistillationCfg: dict = {
    "class_name": "MultiExpertDistillationRunner",
    "student": {
        "class_name": "rsl_rl.modules:MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        "encoder_cfg": {
            "class_name": "rsl_rl.modules:MLP_Encoder",
            "output_detach": False,
            "num_output_dim": 3,
            "hidden_dims": [256, 128],
            "activation": "elu",
            "orthogonal_init": False,
        },
        "encoder_obs_groups": ["obsHistory_flat"],
        "remove_encoder_obs_from_policy": True,
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 0.1,
        },
    },
    "algorithm": {
        "class_name": "MultiExpertDistillation",
        "num_learning_epochs": 4,
        "gradient_length": 12,
        "learning_rate": 1e-3,
        "lr_schedule": "reduce_on_plateau",
        "lr_schedule_factor": 0.5,
        "lr_schedule_patience": 150,
        "lr_schedule_threshold": 5e-4,
        "min_learning_rate": 1e-4,
        "loss_type": "mse",
        "max_grad_norm": 1.0,
        "optimizer": "adam",
        "rollout_action_source": "student",  # fallback when no schedule is configured
        "rollout_action_source_schedule": {
            "mode": "linear_teacher_prob",
            "start_update": 0,
            "end_update": 3000,
            "teacher_prob_start": 1.0,
            "teacher_prob_end": 0.0,
        },
    },
    "obs_groups": {
        "student": ["obsHistory_flat", "policy", "commands", "jump_commands", "gait_commands", "env_group"],
    },
    "experts": [
        {
            "name": "jump_expert",
            "obs_groups": ["obsHistory_flat", "policy", "commands", "jump_commands"],
            "jit_policy_path": _JUMP_JIT_PATH,
            "action_scale": 1.0,
        },
        {
            "name": "gait_expert",
            "obs_groups": ["obsHistory_flat", "policy", "commands", "gait_commands"],
            "jit_policy_path": _GAIT_JIT_PATH,
            "action_scale": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0],
        },re
    ],
    "teacher_id_obs_group": "env_group",
    "num_steps_per_env": 24,
    "save_interval": 500,
    "max_iterations": 20000,
    "experiment_name": "wf_multi_expert_distillation",
}
