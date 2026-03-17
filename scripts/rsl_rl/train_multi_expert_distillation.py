"""训练脚本 — 多专家蒸馏 (MultiExpertDistillationRunner)

用法:
    python train_multi_expert_distillation.py \
        --task Isaac-Limx-WF-MultiExpert-Flat-v0 \
        --num_envs 4096 \
        --headless

教师 JIT 路径在 WF_MultiExpertDistillationCfg 中配置，训练时自动加载，
无需 --teacher_checkpoint。

注意: 学生环境需要提供 "env_group" obs 组 (每个 env 的专家编号)。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="多专家蒸馏训练 (MultiExpertDistillation).")
parser.add_argument("--task", type=str, required=True, help="学生环境 task id.")
parser.add_argument("--num_envs", type=int, default=None, help="并行环境数量.")
parser.add_argument("--max_iterations", type=int, default=None, help="训练迭代次数（覆盖 cfg）.")
parser.add_argument("--video", action="store_true", default=False, help="录制训练视频.")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--video_interval", type=int, default=24000)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import bipedal_locomotion  # noqa: F401 — registers all tasks

from rsl_rl.runner import MultiExpertDistillationRunner
from bipedal_locomotion.tasks.locomotion.agents.limx_rsl_rl_distillation_cfg import (
    WF_MultiExpertDistillationCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # ---- 选择要使用的 config ----
    agent_cfg = copy.deepcopy(WF_MultiExpertDistillationCfg)

    if args_cli.max_iterations is not None:
        agent_cfg["max_iterations"] = args_cli.max_iterations

    # ---- 日志目录 ----
    experiment_name = agent_cfg.get("experiment_name", "multi_expert_distillation")
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))
    log_dir = os.path.join(log_root_path, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print(f"[INFO] Logging to: {log_dir}")

    # ---- 创建学生环境 ----
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env = gym.make(args_cli.task, cfg=env_cfg,
                   render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # ---- 构建 runner（教师 JIT 在 construct_algorithm 内自动加载）----
    runner = MultiExpertDistillationRunner(
        env, agent_cfg, log_dir=log_dir, device=args_cli.device
    )

    # ---- 开始蒸馏训练 ----
    runner.learn(
        num_learning_iterations=agent_cfg.get("max_iterations", 5000),
        init_at_random_ep_len=True,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
