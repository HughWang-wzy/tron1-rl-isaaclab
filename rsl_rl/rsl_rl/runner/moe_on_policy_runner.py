# SPDX-License-Identifier: BSD-3-Clause

"""OnPolicyRunner variant for MoE actor-critic architectures."""

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from rsl_rl.algorithm.moe_ppo import MoEPPO
from rsl_rl.modules import MLP_Encoder
from rsl_rl.modules.moe_actor_critic import MoEActorCritic
from rsl_rl.env import VecEnv


class MoEOnPolicyRunner:
    """Training runner that uses MoEActorCritic + MoEPPO.

    Mirrors ``OnPolicyRunner`` but instantiates MoE-specific modules
    and logs MoE statistics (expert utilization, gate entropy, aux loss).
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.ecd_cfg = train_cfg["encoder"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env

        obs_dict = (self.env.get_observations()).to_dict()
        obs = obs_dict["policy"]
        extras = {"observations": obs_dict}
        self.num_obs = obs.shape[1]
        self.obs_history_len = self.alg_cfg.pop("obs_history_len")
        assert "commands" in extras["observations"]
        self.num_commands = extras["observations"]["commands"].shape[1]
        assert "critic" in extras["observations"]
        num_critic_obs = extras["observations"]["critic"].shape[1] + self.num_commands
        self.ecd_cfg["num_input_dim"] = self.obs_history_len * self.num_obs

        # Encoder (same as OnPolicyRunner)
        encoder = MLP_Encoder(**self.ecd_cfg).to(self.device)

        # MoE Actor-Critic
        num_actor_obs = self.num_obs + encoder.num_output_dim + self.num_commands
        actor_critic = MoEActorCritic(
            num_actor_obs,
            num_critic_obs,
            self.env.num_actions,
            **self.policy_cfg,
        ).to(self.device)

        # MoE PPO algorithm
        # Pop class_name if present (not used as kwarg)
        self.alg_cfg.pop("class_name", None)
        self.alg = MoEPPO(
            self.env.num_envs,
            encoder,
            actor_critic,
            device=self.device,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # Check if expert_target observation group is available
        self.has_expert_target = "expert_target" in obs_dict
        expert_target_shape = None
        if self.has_expert_target:
            expert_target_shape = [obs_dict["expert_target"].shape[1]]

        # Init storage
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.num_obs],
            [num_critic_obs],
            [self.obs_history_len * self.num_obs],
            [self.num_commands],
            [self.env.num_actions],
            expert_target_shape=expert_target_shape,
        )

        self.obs_mean = torch.tensor(0, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_std = torch.tensor(1, dtype=torch.float, device=self.device, requires_grad=False)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "tensorboard")
            self.logger_type = self.logger_type.lower()
            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs_dict = (self.env.get_observations()).to_dict()
        obs = obs_dict["policy"]
        extras = {"observations": obs_dict}
        obs_history = extras["observations"].get("obsHistory").flatten(start_dim=1)
        critic_obs = extras["observations"].get("critic")
        commands = extras["observations"].get("commands")

        obs, obs_history, commands, critic_obs = (
            obs.to(self.device),
            obs_history.to(self.device),
            commands.to(self.device),
            critic_obs.to(self.device),
        )

        # Expert target for current state (aligned with obs used by router)
        expert_target = None
        if self.has_expert_target:
            expert_target = obs_dict["expert_target"].to(self.device)

        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, obs_history, commands, critic_obs)
                    (obs_dict, rewards, dones, infos) = self.env.step(actions)
                    obs_dict = obs_dict.to_dict()
                    obs = obs_dict["policy"]
                    infos["observations"] = obs_dict

                    critic_obs = infos["observations"]["critic"]
                    obs_history = infos["observations"]["obsHistory"].flatten(start_dim=1)
                    commands = infos["observations"]["commands"]

                    obs, obs_history, commands, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        obs_history.to(self.device),
                        commands.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    # Store expert target for MoE router supervision (current state)
                    if self.has_expert_target:
                        self.alg.transition.expert_target = expert_target
                    self.alg.process_env_step(rewards, dones, infos, obs)

                    # Update expert_target for next step
                    if self.has_expert_target:
                        expert_target = infos["observations"]["expert_target"].to(self.device)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
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

                critic_obs_ = torch.cat((critic_obs, commands), dim=-1)
                if self.alg.critic_take_latent:
                    encoder_out = self.alg.encoder.encode(obs_history)
                    self.alg.compute_returns(torch.cat((critic_obs_, encoder_out), dim=-1))
                else:
                    self.alg.compute_returns(critic_obs_)

            (
                mean_value_loss,
                mean_extra_loss,
                mean_surrogate_loss,
                mean_kl,
            ) = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(
            os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration))
        )

    def log(self, locs, width=80, pad=35):
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

        mean_std = torch.exp(self.alg.actor_critic.logstd).mean()
        fps = int(
            self.num_steps_per_env * self.env.num_envs
            / (locs["collection_time"] + locs["learn_time"])
        )

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/encoder", locs["mean_extra_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Policy/mean_kl", locs["mean_kl"], locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])

        # MoE-specific logging
        self.writer.add_scalar("Loss/moe_aux", self.alg.mean_aux_loss, locs["it"])
        self.writer.add_scalar("Loss/router_supervision", self.alg.mean_router_loss, locs["it"])
        self.writer.add_scalar("MoE/gate_entropy", self.alg.mean_gate_entropy, locs["it"])
        if self.alg.expert_utilization is not None:
            for i, util in enumerate(self.alg.expert_utilization):
                self.writer.add_scalar(f"MoE/expert_{i}_utilization", util, locs["it"])

        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str_header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'MoE aux loss:':>{pad}} {self.alg.mean_aux_loss:.4f}\n"""
                f"""{'Router supervision:':>{pad}} {self.alg.mean_router_loss:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.4f}\n"""
                f"""{'Learning rate:':>{pad}} {self.alg.learning_rate:.4f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str_header.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'MoE aux loss:':>{pad}} {self.alg.mean_aux_loss:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        # Expert utilization
        if self.alg.expert_utilization is not None:
            util_str = ", ".join([f"E{i}: {u:.3f}" for i, u in enumerate(self.alg.expert_utilization)])
            log_string += f"""{'Expert utilization:':>{pad}} [{util_str}]\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "encoder_state_dict": self.alg.encoder.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=False):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.encoder.load_state_dict(loaded_dict["encoder_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_encoder(self, device=None):
        self.alg.encoder.eval()
        if device is not None:
            self.alg.encoder.to(device)
        return self.alg.encoder.encode

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
