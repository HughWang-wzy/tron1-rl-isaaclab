# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib
import pkgutil
import torch
import warnings
from tensordict import TensorDict
from typing import Any, Callable


def get_param(param: Any, idx: int) -> Any:
    """Get a parameter for the given index."""
    if isinstance(param, (tuple, list)):
        return param[idx]
    else:
        return param


def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """Resolve the activation function from the name."""
    act_dict = {
        "elu": torch.nn.ELU(),
        "selu": torch.nn.SELU(),
        "relu": torch.nn.ReLU(),
        "crelu": torch.nn.CELU(),
        "lrelu": torch.nn.LeakyReLU(),
        "tanh": torch.nn.Tanh(),
        "sigmoid": torch.nn.Sigmoid(),
        "softplus": torch.nn.Softplus(),
        "gelu": torch.nn.GELU(),
        "swish": torch.nn.SiLU(),
        "mish": torch.nn.Mish(),
        "identity": torch.nn.Identity(),
    }

    act_name = act_name.lower()
    if act_name in act_dict:
        return act_dict[act_name]
    else:
        raise ValueError(f"Invalid activation function '{act_name}'. Valid activations are: {list(act_dict.keys())}")


def resolve_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
    """Resolve the optimizer from the name."""
    optimizer_dict = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    optimizer_name = optimizer_name.lower()
    if optimizer_name in optimizer_dict:
        return optimizer_dict[optimizer_name]
    else:
        raise ValueError(f"Invalid optimizer '{optimizer_name}'. Valid optimizers are: {list(optimizer_dict.keys())}")


def split_and_pad_trajectories(
    tensor: torch.Tensor | TensorDict, dones: torch.Tensor
) -> tuple[torch.Tensor | TensorDict, torch.Tensor]:
    """Split trajectories at done indices and pad to the longest trajectory."""
    dones = dones.clone()
    dones[-1] = 1
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()

    if isinstance(tensor, TensorDict):
        padded_trajectories = {}
        for k, v in tensor.items():
            trajectories = torch.split(v.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
            trajectories = (*trajectories, torch.zeros(v.shape[0], *v.shape[2:], device=v.device))
            padded_trajectories[k] = torch.nn.utils.rnn.pad_sequence(trajectories)
            padded_trajectories[k] = padded_trajectories[k][:, :-1]
        padded_trajectories = TensorDict(
            padded_trajectories, batch_size=[tensor.batch_size[0], len(trajectory_lengths_list)], device=tensor.device
        )
    else:
        trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
        trajectories = (*trajectories, torch.zeros(tensor.shape[0], *tensor.shape[2:], device=tensor.device))
        padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
        padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories: torch.Tensor | TensorDict, masks: torch.Tensor) -> torch.Tensor | TensorDict:
    """Inverse of split_and_pad_trajectories."""
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def string_to_callable(name: str) -> Callable:
    """Resolve the module and function names to return the function."""
    try:
        mod_name, attr_name = name.split(":")
        mod = importlib.import_module(mod_name)
        callable_object = getattr(mod, attr_name)
        if callable(callable_object):
            return callable_object
        else:
            raise ValueError(f"The imported object is not callable: '{name}'")
    except AttributeError as err:
        msg = (
            "We could not interpret the entry as a callable object. The format of input should be"
            f" 'module:attribute_name'\nWhile processing input '{name}'."
        )
        raise ValueError(msg) from err


def resolve_obs_groups(
    obs: TensorDict, obs_groups: dict[str, list[str]], default_sets: list[str]
) -> dict[str, list[str]]:
    """Validate the observation configuration and resolve missing observation sets."""
    # Check if obs_groups is empty
    if len(obs_groups) == 0:
        warnings.warn("The 'obs_groups' dictionary is empty and likely not configured.")
    else:
        for set_name, groups in obs_groups.items():
            if len(groups) == 0:
                raise ValueError(f"The '{set_name}' key in 'obs_groups' cannot be an empty list.")
            for group in groups:
                if group not in obs:
                    raise ValueError(
                        f"Observation '{group}' in set '{set_name}' not found in env observations."
                        f" Available: {list(obs.keys())}"
                    )

    for default_set_name in default_sets:
        if default_set_name not in obs_groups:
            if default_set_name in obs:
                obs_groups[default_set_name] = [default_set_name]
                warnings.warn(
                    f"'{default_set_name}' not in 'obs_groups'. Defaulting to '{default_set_name}' observation group."
                )
            elif "policy" in obs:
                obs_groups[default_set_name] = ["policy"]
                warnings.warn(
                    f"'{default_set_name}' not in 'obs_groups'. Defaulting to 'policy' observations."
                )
            else:
                raise ValueError(
                    f"'{default_set_name}' not in 'obs_groups' and no suitable observation could be found."
                )

    print("-" * 80)
    print("Resolved observation sets: ")
    for set_name, groups in obs_groups.items():
        print("\t", set_name, ": ", groups)
    print("-" * 80)

    return obs_groups


def resolve_callable(callable_or_name: type | Callable | str) -> Callable:
    """Resolve a callable from a string, type, or return it directly.

    Supports:
    - Direct callable: pass a class or function directly.
    - Qualified name with colon: ``"module.path:ClassName"`` (recommended).
    - Qualified name with dot: ``"module.path.ClassName"``.
    - Simple name within rsl_rl: ``"MLPModel"``, ``"GaussianDistribution"``, etc.
    """
    if callable(callable_or_name):
        return callable_or_name

    if not isinstance(callable_or_name, str):
        raise TypeError(f"Expected callable or string, got {type(callable_or_name)}")

    # Qualified name with colon separator
    if ":" in callable_or_name:
        module_path, attr_path = callable_or_name.rsplit(":", 1)
        module = importlib.import_module(module_path)
        obj = module
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
        return obj

    # Qualified name with dot separator
    if "." in callable_or_name:
        parts = callable_or_name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module_path = ".".join(parts[:i])
            attr_parts = parts[i:]
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError:
                continue
            obj = module
            try:
                for attr in attr_parts:
                    obj = getattr(obj, attr)
                return obj
            except AttributeError:
                continue
        raise ImportError(f"Could not resolve '{callable_or_name}': no valid module.attr split found")

    # Simple name — search within rsl_rl submodules
    try:
        import rsl_rl
        for _, module_name, _ in pkgutil.iter_modules(rsl_rl.__path__, "rsl_rl."):
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, callable_or_name):
                    return getattr(module, callable_or_name)
            except Exception:
                continue
    except Exception:
        pass

    raise ValueError(
        f"Could not resolve '{callable_or_name}'. "
        f"Use a qualified name like 'rsl_rl.modules:MLPModel' or pass the class directly."
    )
