# Copyright (c) 2024, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Distillation runner configs for WF locomotion tasks.

Two example configs are provided:

1. WF_JumpDistillationCfg  — 单专家蒸馏
   Student env : Isaac-Limx-WF-Jump-Flat-v0
   Runner class: DistillationRunner (rsl_rl.runner)
   Algorithm   : Distillation  (StudentTeacher behavior cloning)
   Teacher     : loaded from RL checkpoint via runner.load()

2. WF_MultiExpertDistillationCfg  — 多专家蒸馏
   Student env : Isaac-Limx-WF-MultiExpert-Flat-v0  (需要自建，见注释)
   Runner class: MultiExpertDistillationRunner (rsl_rl.runner)
   Algorithm   : MultiExpertDistillation
   Teachers    : JIT-exported .pt files for Jump + Gait experts

Usage example
-------------
>>> from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_distillation_cfg import (
...     WF_JumpDistillationCfg,
...     WF_MultiExpertDistillationCfg,
... )
>>> runner = DistillationRunner(env, WF_JumpDistillationCfg, log_dir, device)
>>> runner.load("/path/to/jump_ppo_checkpoint.pt")  # loads teacher weights
>>> runner.learn(5000)
"""

import os

# ============================================================
# 1. 单专家蒸馏 — Isaac-Limx-WF-Jump-Flat-v0
# ============================================================
#
# 运行流程:
#   1. 用 Isaac-Limx-WF-Jump-Flat-v0 作为学生环境
#   2. 学生 (policy obs) 模仿教师 (critic obs，含特权信息)
#   3. 教师权重从 RL checkpoint 加载:  runner.load("jump_ppo.pt")
#
# 对应环境的 obs 组:
#   policy  → base_ang_vel, proj_gravity, joint_pos, joint_vel, last_action  (dim ≈ 22)
#   critic  → policy obs + base_lin_vel + privileged sensor obs              (dim ≈ 200+)
#
WF_JumpDistillationCfg: dict = {
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


# ============================================================
# 2. 多专家蒸馏 — Jump + Gait 双专家
# ============================================================
#
# 运行流程:
#   1. 需要一个支持 Jump / Gait 两种模式的学生环境
#      该环境须提供 "env_group" obs 组: 形状 (num_envs, 1)
#        env_group[i] = 0  → 第 i 个环境跟随 jump_expert
#        env_group[i] = 1  → 第 i 个环境跟随 gait_expert
#   2. 两位教师均为 JIT (.pt) 文件，在 construct_algorithm 时加载
#   3. 无需手动 runner.load()；教师在构建时直接从文件读取
#
# 学生环境 obs 组要求:
#   policy    → 学生输入 (无特权信息)          (dim ≈ 22)
#   critic    → 教师输入 (含特权信息)           (dim ≈ 200+)
#   env_group → 每个 env 被分配的专家编号       (dim = 1, int 或 one-hot)
#
# 如何添加 env_group obs:
#   在学生环境 cfg 中:
#       self.observations.env_group = ObsGroup(
#           ObsTerm(func=mdp.env_group_id, ...)  # 返回 (num_envs, 1) int tensor
#       )
#
_JUMP_JIT_PATH = os.path.expanduser(
    "~/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_jump/exported/policy.pt"
)
_GAIT_JIT_PATH = os.path.expanduser(
    "~/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/exported/policy.pt"
)

WF_MultiExpertDistillationCfg: dict = {
    # ---- 学生模型 (MLPModel) ----
    "student": {
        "class_name": "rsl_rl.modules:MLPModel",
        "hidden_dims": [512, 256, 128],
        "activation": "elu",
        "obs_normalization": False,
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
        "student": ["policy"],      # 学生只看策略 obs (无特权)
    },
    # ---- 专家列表 ----
    #   每位专家:
    #     name            → 在日志/指标中的标识符
    #     obs_groups      → 教师 JIT 模型的输入 obs 组 (拼接后送入 JIT)
    #     jit_policy_path → TorchScript (.pt) 文件路径
    "experts": [
        {
            "name": "jump_expert",
            "obs_groups": ["policy", "critic"],   # 教师看 policy+critic (与训练时一致)
            "jit_policy_path": _JUMP_JIT_PATH,
        },
        {
            "name": "gait_expert",
            "obs_groups": ["policy", "critic"],
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
