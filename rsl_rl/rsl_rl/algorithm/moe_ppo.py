# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import MLP_Encoder
from rsl_rl.modules.moe_actor_critic import MoEActorCritic
from rsl_rl.storage import RolloutStorage


def compute_moe_aux_loss(gate_probs, num_experts):
    """Load-balancing auxiliary loss.

    Encourages the gating network to distribute load evenly across experts
    by minimising negative entropy of mean expert weights plus mean absolute
    deviation from uniform.

    Args:
        gate_probs: (batch, num_experts) softmax probabilities.
        num_experts: number of experts.
    Returns:
        Scalar loss tensor.
    """
    eps = 1e-8
    w_bar = gate_probs.mean(dim=0)  # (num_experts,)
    # Negative entropy: encourages uniform distribution
    entropy_term = -torch.sum(w_bar * torch.log(w_bar + eps))
    # MAE from uniform
    target_prob = 1.0 / num_experts
    mae_term = 0.5 * torch.sum(torch.abs(w_bar - target_prob))
    return entropy_term + mae_term


class MoEPPO:
    """PPO with Mixture-of-Experts actor and load-balancing auxiliary loss."""

    actor_critic: MoEActorCritic
    encoder: MLP_Encoder

    def __init__(
        self,
        num_group,
        encoder,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        vae_beta=1.0,
        est_learning_rate=1.0e-3,
        critic_take_latent=False,
        early_stop=False,
        anneal_lr=False,
        moe_aux_loss_coef=0.1,
        device="cpu",
        **kwargs,
    ):
        self.device = device
        self.num_group = num_group

        self.desired_kl = desired_kl
        self.early_stop = early_stop
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.anneal_lr = anneal_lr
        self.vae_beta = vae_beta
        self.critic_take_latent = critic_take_latent
        self.moe_aux_loss_coef = moe_aux_loss_coef

        self.encoder = encoder

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(
            [{"params": self.actor_critic.parameters()}], lr=learning_rate
        )

        if self.encoder.num_output_dim != 0:
            self.extra_optimizer = optim.Adam(
                self.encoder.parameters(), lr=est_learning_rate
            )
        else:
            self.extra_optimizer = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # MoE tracking
        self.mean_aux_loss = 0.0
        self.mean_gate_entropy = 0.0
        self.expert_utilization = None

    def init_storage(
        self,
        num_envs,
        num_transitions_per_env,
        actor_obs_shape,
        critic_obs_shape,
        obs_history_shape,
        commands_shape,
        action_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            obs_history_shape,
            commands_shape,
            action_shape,
            self.device,
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_history, commands, critic_obs):
        critic_obs = torch.cat((critic_obs, commands), dim=-1)
        encoder_out = self.encoder.encode(obs_history)
        self.transition.actions = self.actor_critic.act(
            torch.cat((encoder_out, obs, commands), dim=-1)
        ).detach()

        if self.critic_take_latent:
            critic_obs = torch.cat((critic_obs, encoder_out), dim=-1)
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()

        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(
            self.transition.actions
        ).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_obs = critic_obs
        self.transition.observation_history = obs_history
        self.transition.commands = commands
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, next_obs=None):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values
                * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.transition.next_observations = next_obs
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        num_updates = 0
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_kl = 0
        mean_aux_loss = 0
        total_gate_probs = None

        generator = self.storage.mini_batch_generator(
            self.num_group,
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        for (
            obs_batch,
            critic_obs_batch,
            obs_history_batch, _,
            group_commands_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
        ) in generator:
            encoder_out_batch = self.encoder.encode(obs_history_batch)
            commands_batch = group_commands_batch
            self.actor_critic.act(
                torch.cat(
                    (encoder_out_batch, obs_batch, commands_batch),
                    dim=-1,
                )
            )

            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(
                actions_batch
            )

            value_batch = self.actor_critic.evaluate(critic_obs_batch)
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL divergence
            kl_mean = torch.tensor(0, device=self.device, requires_grad=False)
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                    + (
                        torch.square(old_sigma_batch)
                        + torch.square(old_mu_batch - mu_batch)
                    )
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

            # Adaptive LR
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.desired_kl is not None and self.early_stop:
                if kl_mean > self.desired_kl * 1.5:
                    break

            # Surrogate loss
            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_batch_mean = entropy_batch.mean()

            # MoE auxiliary loss
            gate_probs = self.actor_critic.gate_probs  # (batch, num_experts)
            aux_loss = compute_moe_aux_loss(gate_probs, self.actor_critic.num_experts)

            # Track gate statistics
            with torch.inference_mode():
                if total_gate_probs is None:
                    total_gate_probs = gate_probs.mean(dim=0).detach()
                else:
                    total_gate_probs += gate_probs.mean(dim=0).detach()

            # Total loss
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch_mean
                + self.moe_aux_loss_coef * aux_loss
            )

            if self.anneal_lr:
                frac = 1.0 - num_updates / (
                    self.num_learning_epochs * self.num_mini_batches
                )
                self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            num_updates += 1
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_kl += kl_mean.item()
            mean_aux_loss += aux_loss.item()

        # Encoder update (same as PPO)
        num_updates_extra = 0
        mean_extra_loss = 0
        if self.extra_optimizer is not None:
            generator = self.storage.encoder_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
            for (
                next_obs_batch,
                critic_obs_batch,
                obs_history_batch,
            ) in generator:
                if self.encoder.is_mlp_encoder:
                    self.encoder.encode(obs_history_batch)
                    encode_batch = self.encoder.get_encoder_out()

                if self.encoder.is_mlp_encoder:
                    extra_loss = (
                        (encode_batch[:, 0:3] - critic_obs_batch[:, 0:3]).pow(2).mean()
                    )
                else:
                    extra_loss = torch.zeros(1, device=self.device)

                self.extra_optimizer.zero_grad()
                extra_loss.backward()
                self.extra_optimizer.step()

                num_updates_extra += 1
                mean_extra_loss += extra_loss.item()

        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_kl /= num_updates
            mean_aux_loss /= num_updates
            total_gate_probs /= num_updates
        if num_updates_extra > 0:
            mean_extra_loss /= num_updates_extra

        # Store MoE statistics for logging
        self.mean_aux_loss = mean_aux_loss
        if total_gate_probs is not None:
            self.expert_utilization = total_gate_probs.cpu().numpy()
            eps = 1e-8
            self.mean_gate_entropy = -(
                total_gate_probs * torch.log(total_gate_probs + eps)
            ).sum().item()

        self.storage.clear()

        return (mean_value_loss, mean_extra_loss, mean_surrogate_loss, mean_kl)
