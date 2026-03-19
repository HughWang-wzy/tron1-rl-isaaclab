
import os

WF_JumpDistillationCfg: dict = {
    "class_name": "DistillationRunner",
    # ---- 学生-教师策略 ----
    "policy": {
        "class_name": "StudentTeacher",
        "student_hidden_dims": [512, 256, 128],
        "teacher_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "student_obs_normalization": False,
        "teacher_obs_normalization": False,
    },
    # ---- 蒸馏算法 ----
    "algorithm": {
        "class_name": "Distillation",
        "num_learning_epochs": 1,
        "gradient_length": 15,      # 每隔 gradient_length 个时间步做一次反向传播
        "learning_rate": 1e-3,
        "loss_type": "mse",         # 或 "huber"
        "max_grad_norm": 1.0,
    },
    # ---- 观测组分配 ----
    #   "policy"  → 学生输入
    #   "teacher" → 教师输入 (特权信息)
    "obs_groups": {
        "policy":  ["policy"],   # 学生只看策略 obs
        "teacher": ["critic"],   # 教师看特权 obs
    },
    # ---- 训练超参数 ----
    "num_steps_per_env": 24,
    "save_interval": 200,
    "max_iterations": 5000,
    "experiment_name": "wf_jump_distillation",
}

_JUMP_JIT_PATH = os.path.expanduser(
    "/home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_jump/2026-03-17_23-42-13-rough/exported/policy_all.pt"
)
_GAIT_JIT_PATH = os.path.expanduser(
    "/home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/policy_all.pt"
)

WF_MultiExpertDistillationCfg: dict = {
    "class_name": "MultiExpertDistillationRunner",
    # ---- 学生模型 (MLPModel) ----
    "student": {
        "class_name": "rsl_rl.modules:MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
        # 复用 PPO 的 EncoderCfg 风格：student 先编码 obsHistory_flat，再与其余 obs 拼接进 policy MLP
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
        # 高斯分布输出: 均值由网络给出, 标准差可学习
        "distribution_cfg": {
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
        },
    },
    # ---- 多专家蒸馏算法 ----
    "algorithm": {
        "class_name": "MultiExpertDistillation",
        "num_learning_epochs": 1,
        "gradient_length": 15,
        "learning_rate": 1e-3,
        "loss_type": "mse",         # 或 "huber"
        "max_grad_norm": 1.0,
        "optimizer": "adam",
    },
    # ---- 学生观测组 ----
    "obs_groups": {
        # 学生需要命令条件 + 历史信息，否则在多任务目标下会学成“平均动作”
        "student": ["obsHistory_flat", "policy", "commands", "gait_commands", "env_group"],
    },
    # ---- 专家列表 ----
    #   每位专家:
    #     name            → 在日志/指标中的标识符
    #     obs_groups      → 教师 JIT 模型的输入 obs 组 (拼接后送入 JIT)
    #     jit_policy_path → TorchScript (.pt) 文件路径
    "experts": [
        {
            "name": "jump_expert",
            # x = [obsHistory_flat, policy, commands]
            # policy_all 内部执行: encoder(obsHistory_flat) -> concat -> actor
            "obs_groups": ["obsHistory_flat", "policy", "commands"],
            "jit_policy_path": _JUMP_JIT_PATH,
        },
        {
            "name": "gait_expert",
            # x = [obsHistory_flat, policy, commands, gait_commands]
            # policy_all 内部将 commands[:3] 与 gait_commands 合成为 gait actor 所需 commands
            "obs_groups": ["obsHistory_flat", "policy", "commands", "gait_commands"],
            "jit_policy_path": _GAIT_JIT_PATH,
        },
    ],
    # ---- 路由观测组 ----
    #   该 obs 组须在学生环境中存在，用于决定每个 env 跟随哪位专家:
    #     标量整数: shape (num_envs,)  或  (num_envs, 1)
    #     one-hot : shape (num_envs, num_experts)
    "teacher_id_obs_group": "env_group",
    # ---- 训练超参数 ----
    "num_steps_per_env": 24,
    "save_interval": 200,
    "max_iterations": 5000,
    "experiment_name": "wf_multi_expert_distillation",
}
