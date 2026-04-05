# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import os
import time
from collections import deque
import statistics

import torch
from torch.utils.tensorboard import SummaryWriter

from rsl_rl.algorithm import MultiExpertDistillation


class MultiExpertDistillationRunner:
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.device = device
        self.env = env

        self.num_steps_per_env: int = train_cfg["num_steps_per_env"]
        self.save_interval: int = train_cfg.get("save_interval", 100)

        # Query initial observations from the environment (returns TensorDict)
        obs = self.env.get_observations()

        # Deep-copy cfg so construct_algorithm's pop() calls don't mutate train_cfg
        alg_cfg = copy.deepcopy(train_cfg)

        # MultiExpertDistillation.construct_algorithm builds student, teachers, storage, and alg
        self.alg: MultiExpertDistillation = MultiExpertDistillation.construct_algorithm(
            obs, env, alg_cfg, device
        )

        # Logging
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        self.env.reset()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run multi-expert distillation training.

        Args:
            num_learning_iterations: Number of training iterations.
            init_at_random_ep_len: Randomize initial episode lengths.

        Raises:
            ValueError: If teacher models have not been successfully loaded.
        """
        if not self.alg.teacher_loaded:
            raise ValueError(
                "Teacher models not loaded. Ensure all experts' jit_policy_path are valid "
                "and construct_algorithm() completed successfully."
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

        self.alg.train_mode()

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
                    # print(f"obs: {obs}")
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

        loss_dict: dict = locs["loss_dict"]
        fps = int(
            self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"])
        )
        behavior_loss = loss_dict.get("behavior", 0.0)

        self.writer.add_scalar("Loss/behavior", behavior_loss, locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        # Log per-expert losses
        for key, value in loss_dict.items():
            if key.startswith("behavior_"):
                self.writer.add_scalar(f"Loss/{key}", value, locs["it"])

        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection_time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar(
                "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"]
            )

        # Build per-expert loss display string
        expert_loss_string = ""
        for key, value in loss_dict.items():
            if key.startswith("behavior_"):
                expert_name = key[len("behavior_"):]
                expert_loss_string += f"""{f'Loss [{expert_name}]:':>{pad}} {value:.4f}\n"""

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
                f"""{'Learning rate:':>{pad}} {self.alg.learning_rate:.6f}\n"""
                f"""{expert_loss_string}"""
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
                f"""{'Learning rate:':>{pad}} {self.alg.learning_rate:.6f}\n"""
                f"""{expert_loss_string}"""
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
        state = self.alg.save()
        state["iter"] = self.current_learning_iteration
        state["infos"] = infos
        torch.save(state, path)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
    ) -> dict | None:
        """Load a checkpoint.

        Args:
            path: Path to the checkpoint file.
            load_cfg: Optional dict controlling what to load:
                ``{"student": True, "optimizer": True, "iteration": True}``.
                If ``None``, the algorithm infers defaults from the checkpoint.
            strict: Whether to use strict ``load_state_dict`` for the student.

        Returns:
            The ``infos`` field stored in the checkpoint, or None.
        """
        loaded_dict = torch.load(path, map_location=self.device)
        restore_iter = self.alg.load(loaded_dict, load_cfg, strict)
        if restore_iter and "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None):
        """Return the student policy in inference mode."""
        student = self.alg.get_policy()
        student.eval()
        if device is not None:
            student.to(device)
        return student
