# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MoEActorCritic(nn.Module):
    """Actor-Critic with Mixture-of-Experts actor network.

    The actor uses a gating network to route inputs to N expert MLPs.
    Top-K experts are selected per sample and their outputs are combined
    via weighted sum.  The critic remains a standard MLP.

    Stores ``gate_probs`` (shape: batch × num_experts) after each forward
    pass so the algorithm can compute the load-balancing auxiliary loss.
    """

    is_recurrent = False
    is_sequence = False
    is_vae = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        num_experts=2,
        top_k=1,
        gating_hidden_dims=[128, 64],
        expert_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "MoEActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.num_experts = num_experts
        self.top_k = top_k

        act_fn = _get_activation(activation)

        # ---- Gating network ----
        gating_layers = []
        prev_dim = num_actor_obs
        for dim in gating_hidden_dims:
            gating_layers.append(nn.Linear(prev_dim, dim))
            gating_layers.append(_get_activation(activation))
            prev_dim = dim
        gating_layers.append(nn.Linear(prev_dim, num_experts))
        self.gating_network = nn.Sequential(*gating_layers)

        # ---- Expert networks ----
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            layers = []
            prev_dim = num_actor_obs
            for i, dim in enumerate(expert_hidden_dims):
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(_get_activation(activation))
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, num_actions))
            self.experts.append(nn.Sequential(*layers))

        # ---- Critic (standard MLP) ----
        critic_layers = []
        prev_dim = num_critic_obs
        for i, dim in enumerate(critic_hidden_dims):
            critic_layers.append(nn.Linear(prev_dim, dim))
            critic_layers.append(_get_activation(activation))
            prev_dim = dim
        critic_layers.append(nn.Linear(prev_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # ---- Action noise ----
        self.logstd = nn.Parameter(torch.zeros(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

        # ---- Stored for aux loss ----
        self.gate_probs = None

        print(f"MoE Actor: {num_experts} experts, top_k={top_k}")
        print(f"  Gating: {self.gating_network}")
        print(f"  Expert[0]: {self.experts[0]}")
        print(f"Critic MLP: {self.critic}")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    # ---- Actor (MoE) forward ----

    def _moe_forward(self, observations):
        """Run the MoE actor and return action means.

        Also stores ``self.gate_probs`` for the auxiliary loss.
        """
        # Gate probabilities
        gate_logits = self.gating_network(observations)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (batch, num_experts)
        self.gate_probs = gate_probs

        # Top-K selection
        topk_weights, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # Re-normalise selected weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        batch_size = observations.shape[0]
        action_mean = torch.zeros(batch_size, self.num_actions, device=observations.device)

        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # (batch,)
            weight = topk_weights[:, i].unsqueeze(1)  # (batch, 1)
            for e_idx in range(self.num_experts):
                mask = expert_idx == e_idx
                if mask.any():
                    expert_out = self.experts[e_idx](observations[mask])
                    action_mean[mask] += weight[mask] * expert_out

        return action_mean

    # ---- Interface matching ActorCritic ----

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self._moe_forward(observations)
        self.distribution = Normal(mean, mean * 0.0 + torch.exp(self.logstd))

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        return self._moe_forward(observations)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)


def _get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print(f"invalid activation function: {act_name}")
        return nn.ELU()
