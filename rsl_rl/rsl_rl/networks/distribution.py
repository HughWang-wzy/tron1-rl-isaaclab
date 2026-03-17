# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class Distribution(nn.Module):
    """Base class for distribution modules used by MLPModel."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        raise NotImplementedError

    def sample(self) -> torch.Tensor:
        raise NotImplementedError

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def as_deterministic_output_module(self) -> nn.Module:
        raise NotImplementedError

    @property
    def input_dim(self) -> int | list[int]:
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def entropy(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        pass


class GaussianDistribution(Distribution):
    """Gaussian distribution with state-independent (learnable) standard deviation."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
    ) -> None:
        super().__init__(output_dim)
        self.std_type = std_type

        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)))
        else:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        mean = mlp_output
        if self.std_type == "scalar":
            std = self.std_param.expand_as(mean)
        elif self.std_type == "log":
            std = torch.exp(self.log_std_param).expand_as(mean)
        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        return _IdentityDeterministicOutput()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.mean

    @property
    def std(self) -> torch.Tensor:
        return self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(outputs).sum(dim=-1)

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)


class HeteroscedasticGaussianDistribution(GaussianDistribution):
    """Gaussian distribution with state-dependent (MLP-output) standard deviation."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_type: str = "scalar",
    ) -> None:
        Distribution.__init__(self, output_dim)
        self.std_type = std_type
        self.init_std = init_std

        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
        elif self.std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]

    def as_deterministic_output_module(self) -> nn.Module:
        return _MeanSliceDeterministicOutput()

    @property
    def input_dim(self) -> list[int]:
        return [2, self.output_dim]

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim:])
        if self.std_type == "scalar":
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim:], self.init_std)
        elif self.std_type == "log":
            init_std_log = torch.log(torch.tensor(self.init_std + 1e-7))
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim:], init_std_log)


class _IdentityDeterministicOutput(nn.Module):
    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output


class _MeanSliceDeterministicOutput(nn.Module):
    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]
