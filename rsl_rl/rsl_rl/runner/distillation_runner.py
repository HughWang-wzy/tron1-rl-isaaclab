# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
from collections import deque
import statistics

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithm import Distillation
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from rsl_rl.storage import DistillationRolloutStorage


class DistillationRunner:
    """Distillation runner for training student-teacher models.

    This runner follows the same interface as tron1's OnPolicyRunner but is
    specialized for behavior-cloning distillation from a teacher to a student policy.

    Expected train_cfg structure::

        train_cfg = {
            "policy": {
                "class_name": "StudentTeacher",  # or "StudentTeacherRecurrent"
                "student_hidden_dims": [512, 256, 128],
                "teacher_hidden_dims": [512, 256, 128],
                # ... other StudentTeacher kwargs
            },
            "algorithm": {
                "class_name": "Distillation",
                "num_learning_epochs": 1,
                "gradient_length": 15,
                "learning_rate": 1e-3,
                "loss_type": "mse",
                # ... other Distillation kwargs
            },
            "obs_groups": {
                "policy": ["policy"],       # observation keys used by student
                "teacher": ["critic"],      # observation keys used by teacher (privileged)
            },
            "num_steps_per_env": 24,
            "save_interval": 100,
        }

    Usage::

        runner = DistillationRunner(env, train_cfg, log_dir, device)
        runner.load("teacher_checkpoint.pt")   # load teacher weights
        runner.learn(num_learning_iterations)
    """

    _POLICY_CLASSES = {
        "StudentTeacher": StudentTeacher,
        "StudentTeacherRecurrent": StudentTeacherRecurrent,
    }
    _ALG_CLASSES = {
        "Distillation": Distillation,
    }

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        # Copy mutable dicts to avoid modifying the original config
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.policy_cfg = dict(train_cfg["policy"])
        self.device = device
        self.env = env

        # Query initial observations from the environment (returns TensorDict)
        obs = self.env.get_observations()

        # --- Build student-teacher policy ---
        policy_class_name = self.policy_cfg.pop("class_name")
        if policy_class_name not in self._POLICY_CLASSES:
            raise ValueError(
                f"Unknown policy class '{policy_class_name}'. "
                f"Supported: {list(self._POLICY_CLASSES.keys())}"
            )
        student_teacher_class = self._POLICY_CLASSES[policy_class_name]
        student_teacher = student_teacher_class(
            obs, train_cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(device)

        # --- Build distillation storage ---
        storage = DistillationRolloutStorage(
            "distillation",
            self.env.num_envs,
            train_cfg["num_steps_per_env"],
            obs,
            [self.env.num_actions],
            device,
        )

        # --- Build distillation algorithm ---
        alg_class_name = self.alg_cfg.pop("class_name")
        if alg_class_name not in self._ALG_CLASSES:
            raise ValueError(
                f"Unknown algorithm class '{alg_class_name}'. "
                f"Supported: {list(self._ALG_CLASSES.keys())}"
            )
        alg_class = self._ALG_CLASSES[alg_class_name]
        self.alg: Distillation = alg_class(
            student_teacher, storage, device=device, **self.alg_cfg
        )

        self.num_steps_per_env: int = train_cfg["num_steps_per_env"]
        self.save_interval: int = train_cfg["save_interval"]

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run distillation training.

        Args:
            num_learning_iterations: Number of training iterations.
            init_at_random_ep_len: Randomize initial episode lengths for exploration.

        Raises:
            ValueError: If teacher weights have not been loaded via :meth:`load`.
        """
        if not self.alg.policy.loaded_teacher:
            raise ValueError(
                "Teacher model parameters not loaded. "
                "Please call runner.load('teacher_checkpoint.pt') before training."
            )

        # Initialize tensorboard writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        obs = obs.to(self.device)

        self.alg.policy.train()  # student in train mode, teacher stays in eval

        ep_infos = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_it = self.current_learning_iteration
        tot_iter = start_it + num_learning_iterations

        for it in range(start_it, tot_iter):
            start = time.time()

            # --- Rollout phase ---
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start
            start = stop

            # --- Learning phase ---
            self.alg.compute_returns(obs)  # no-op for distillation
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        """Log training progress to tensorboard and console."""
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""

        fps = int(
            self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"])
        )
        behavior_loss = locs["loss_dict"].get("behavior", 0.0)

        self.writer.add_scalar("Loss/behavior", behavior_loss, locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar(
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"]
            )

        str_header = (
            f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        )
        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s """
                f"""(collection: {locs['collection_time']:.3f}s, learning: {locs['learn_time']:.3f}s)\n"""
                f"""{'Behavior loss:':>{pad}} {behavior_loss:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s """
                f"""(collection: {locs['collection_time']:.3f}s, learning: {locs['learn_time']:.3f}s)\n"""
                f"""{'Behavior loss:':>{pad}} {behavior_loss:.4f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path: str, load_optimizer: bool = False) -> dict | None:
        """Load a checkpoint.

        Supports loading from:
        - RL training checkpoints (containing ``actor`` keys) — loads teacher weights.
        - Distillation checkpoints (containing ``student`` keys) — resumes training.

        Args:
            path: Path to the checkpoint file.
            load_optimizer: Whether to restore optimizer state (only for distillation checkpoints).

        Returns:
            The ``infos`` field stored in the checkpoint, or None.
        """
        loaded_dict = torch.load(path, map_location=self.device)
        # StudentTeacher.load_state_dict handles both RL and distillation checkpoints
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None):
        """Return student policy in inference mode."""
        self.alg.policy.eval()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.policy.act_inference
