# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.networks import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.networks.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class MLPModel(nn.Module):
    """MLP-based neural model for multi-expert distillation.

    Processes 1D observation groups via a multi-layer perceptron. Supports optional
    observation normalization and stochastic output via a configurable distribution module.

    Used as the trainable student in :class:`MultiExpertDistillation`.
    """

    is_recurrent: bool = False

    @staticmethod
    def _obs_group_tensor(obs: TensorDict, obs_group: str) -> torch.Tensor:
        """Fetch one observation group and flatten history-style [N, H, D] tensors."""
        tensor = obs[obs_group]
        if tensor.ndim == 3:
            return tensor.flatten(start_dim=1)
        if tensor.ndim == 2:
            return tensor
        raise ValueError(
            f"MLPModel supports 2D/3D observations only, got shape {tuple(tensor.shape)} for '{obs_group}'."
        )

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict | None = None,
        encoder_obs_groups: list[str] | None = None,
        remove_encoder_obs_from_policy: bool = True,
    ) -> None:
        super().__init__()

        active_obs_groups = list(obs_groups[obs_set])
        self.encoder = None
        self.encoder_obs_groups: list[str] = []
        self.policy_obs_groups: list[str] = active_obs_groups
        self.encoder_out_dim = 0
        self.encoder_obs_dim = 0

        if encoder_cfg is not None:
            encoder_cfg = dict(encoder_cfg)
            self.encoder_obs_groups = (
                list(encoder_obs_groups)
                if encoder_obs_groups is not None
                else (["obsHistory_flat"] if "obsHistory_flat" in active_obs_groups else [])
            )
            if len(self.encoder_obs_groups) == 0:
                raise ValueError(
                    "encoder_cfg is set but encoder_obs_groups is empty. "
                    "Set encoder_obs_groups explicitly or include 'obsHistory_flat' in student obs groups."
                )
            encoder_input_dim = self._sum_obs_dim(obs, self.encoder_obs_groups)
            self.encoder_obs_dim = encoder_input_dim
            encoder_class = resolve_callable(encoder_cfg.pop("class_name", "MLP_Encoder"))
            encoder_cfg["num_input_dim"] = encoder_input_dim
            self.encoder = encoder_class(**encoder_cfg)
            self.encoder_out_dim = int(self.encoder.num_output_dim)

            if remove_encoder_obs_from_policy:
                self.policy_obs_groups = [g for g in active_obs_groups if g not in self.encoder_obs_groups]
                if len(self.policy_obs_groups) == 0:
                    raise ValueError(
                        "All student obs groups are consumed by encoder; "
                        "please keep at least one policy obs group or set remove_encoder_obs_from_policy=False."
                    )

        self.obs_groups = self.policy_obs_groups
        self.policy_obs_dim = self._sum_obs_dim(obs, self.policy_obs_groups)
        self.obs_dim = self.policy_obs_dim + self.encoder_out_dim

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        if distribution_cfg is not None:
            dist_cfg = dict(distribution_cfg)
            dist_class: type[Distribution] = resolve_callable(dist_cfg.pop("class_name"))
            self.distribution: Distribution | None = dist_class(output_dim, **dist_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        self.mlp = MLP(self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)

        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        mlp_output = self.mlp(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        del masks, hidden_state
        obs_list = [self._obs_group_tensor(obs, obs_group) for obs_group in self.policy_obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        if self.encoder is not None:
            encoder_obs_list = [self._obs_group_tensor(obs, obs_group) for obs_group in self.encoder_obs_groups]
            encoder_obs = torch.cat(encoder_obs_list, dim=-1)
            encoder_out = self.encoder.encode(encoder_obs)
            latent = torch.cat((encoder_out, latent), dim=-1)
        latent = self.obs_normalizer(latent)
        return latent

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(self.get_latent(obs))

    def as_jit(self) -> nn.Module:
        """Return a JIT-exportable copy of this model."""
        return _TorchMLPModel(self)

    def _sum_obs_dim(self, obs: TensorDict, active_obs_groups: list[str]) -> int:
        obs_dim = 0
        for obs_group in active_obs_groups:
            obs_tensor = obs[obs_group]
            if obs_tensor.ndim == 2:
                obs_dim += int(obs_tensor.shape[-1])
            elif obs_tensor.ndim == 3:
                obs_dim += int(obs_tensor.shape[1] * obs_tensor.shape[2])
            else:
                raise ValueError(
                    f"MLPModel supports 2D/3D observations only, got shape {tuple(obs_tensor.shape)} for '{obs_group}'."
                )
        return obs_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim


class _TorchMLPModel(nn.Module):
    """JIT-exportable version of MLPModel (deterministic inference only)."""

    def __init__(self, model: MLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder) if model.encoder is not None else None
        self.mlp = copy.deepcopy(model.mlp)
        self.encoder_obs_dim = model.encoder_obs_dim
        self.policy_obs_dim = model.policy_obs_dim
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoder is not None:
            encoder_obs, policy_obs = torch.split(
                x, [self.encoder_obs_dim, self.policy_obs_dim], dim=-1
            )
            encoder_out = self.encoder(encoder_obs)
            x = torch.cat((encoder_out, policy_obs), dim=-1)
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        pass
