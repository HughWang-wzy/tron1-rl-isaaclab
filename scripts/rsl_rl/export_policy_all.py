#!/usr/bin/env python3
"""Export fused teacher policy (encoder + actor) as TorchScript.

This script builds a single ``policy_all.pt`` from an on-policy checkpoint
(``model_*.pt``). The exported model embeds:

1) encoder(obsHistory_flat)
2) actor input assembly
3) actor forward

So multi-expert distillation can load one JIT file per expert.
"""

from __future__ import annotations

import argparse
import os
import re

import torch
import torch.nn as nn


def _collect_layer_indices(state_dict: dict, prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.weight$")
    indices: list[int] = []
    for key in state_dict:
        match = pattern.match(key)
        if match is not None:
            indices.append(int(match.group(1)))
    indices.sort()
    if len(indices) == 0:
        raise ValueError(f"No layers found for prefix '{prefix}'.")
    return indices


def _build_mlp_from_state_dict(state_dict: dict, prefix: str) -> nn.Sequential:
    indices = _collect_layer_indices(state_dict, prefix)
    modules: list[nn.Module] = []
    for layer_id, layer_index in enumerate(indices):
        weight_key = f"{prefix}{layer_index}.weight"
        bias_key = f"{prefix}{layer_index}.bias"
        weight = state_dict[weight_key]
        bias = state_dict[bias_key]

        layer = nn.Linear(weight.shape[1], weight.shape[0], bias=True)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        modules.append(layer)
        if layer_id < len(indices) - 1:
            modules.append(nn.ELU())
    return nn.Sequential(*modules)


def _first_linear_in_dim(mlp: nn.Sequential) -> int:
    for module in mlp:
        if isinstance(module, nn.Linear):
            return module.in_features
    raise ValueError("MLP has no Linear layer.")


def _last_linear_out_dim(mlp: nn.Sequential) -> int:
    for module in reversed(mlp):
        if isinstance(module, nn.Linear):
            return module.out_features
    raise ValueError("MLP has no Linear layer.")


class JumpPolicyAll(nn.Module):
    """x = [obsHistory_flat, policy, commands] -> actions."""

    def __init__(
        self,
        encoder: nn.Sequential,
        actor: nn.Sequential,
        obs_history_dim: int,
        policy_dim: int,
        commands_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.obs_history_dim = obs_history_dim
        self.policy_dim = policy_dim
        self.commands_dim = commands_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs_history, policy, commands = torch.split(
            x, [self.obs_history_dim, self.policy_dim, self.commands_dim], dim=-1
        )
        encoder_out = self.encoder(obs_history)
        actor_in = torch.cat((encoder_out, policy, commands), dim=-1)
        return self.actor(actor_in)

    @torch.jit.export
    def reset(self) -> None:
        pass


class GaitPolicyAll(nn.Module):
    """x = [obsHistory_flat, policy, commands, gait_commands] -> actions.

    ``commands`` from the current student env is:
      [base_velocity(3), base_jump(3)]
    Gait actor expects commands:
      [base_velocity(3), gait_command(4), standing_height(1)]

    So we keep only ``commands[:3]`` and append ``gait_commands``.
    """

    def __init__(
        self,
        encoder: nn.Sequential,
        actor: nn.Sequential,
        obs_history_dim: int,
        policy_dim: int,
        commands_dim: int,
        gait_commands_dim: int,
        velocity_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.obs_history_dim = obs_history_dim
        self.policy_dim = policy_dim
        self.commands_dim = commands_dim
        self.gait_commands_dim = gait_commands_dim
        self.velocity_dim = velocity_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs_history, policy, commands, gait_commands = torch.split(
            x,
            [
                self.obs_history_dim,
                self.policy_dim,
                self.commands_dim,
                self.gait_commands_dim,
            ],
            dim=-1,
        )
        velocity_commands = commands[..., : self.velocity_dim]
        actor_commands = torch.cat((velocity_commands, gait_commands), dim=-1)
        encoder_out = self.encoder(obs_history)
        actor_in = torch.cat((encoder_out, policy, actor_commands), dim=-1)
        return self.actor(actor_in)

    @torch.jit.export
    def reset(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fused policy_all.pt from RSL-RL checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt checkpoint.")
    parser.add_argument("--mode", type=str, choices=["jump", "gait"], required=True)
    parser.add_argument("--output", type=str, default=None, help="Output path for policy_all.pt.")
    parser.add_argument("--obs-history-dim", type=int, default=None, help="Override obsHistory_flat dim.")
    parser.add_argument("--policy-dim", type=int, default=28, help="Policy-group dimension.")
    parser.add_argument("--commands-dim", type=int, default=6, help="Commands-group dimension.")
    parser.add_argument("--gait-commands-dim", type=int, default=5, help="Gait-commands-group dimension.")
    parser.add_argument("--velocity-dim", type=int, default=3, help="Velocity dims taken from commands for gait.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_state_dict = checkpoint.get("model_state_dict")
    encoder_state_dict = checkpoint.get("encoder_state_dict")
    if model_state_dict is None or encoder_state_dict is None:
        raise KeyError(
            "Checkpoint must contain both 'model_state_dict' and 'encoder_state_dict'."
        )

    actor = _build_mlp_from_state_dict(model_state_dict, "actor.")
    encoder = _build_mlp_from_state_dict(encoder_state_dict, "encoder.")
    actor.eval()
    encoder.eval()

    actor_input_dim = _first_linear_in_dim(actor)
    actor_output_dim = _last_linear_out_dim(actor)
    encoder_input_dim = _first_linear_in_dim(encoder)
    encoder_output_dim = _last_linear_out_dim(encoder)

    obs_history_dim = encoder_input_dim if args.obs_history_dim is None else args.obs_history_dim
    if obs_history_dim != encoder_input_dim:
        raise ValueError(
            f"obs_history_dim mismatch: provided {obs_history_dim}, encoder expects {encoder_input_dim}."
        )

    policy_dim = args.policy_dim
    commands_dim = args.commands_dim
    gait_commands_dim = args.gait_commands_dim
    velocity_dim = args.velocity_dim

    if args.mode == "jump":
        expected_actor_in = encoder_output_dim + policy_dim + commands_dim
        if actor_input_dim != expected_actor_in:
            raise ValueError(
                f"Jump actor input mismatch: actor expects {actor_input_dim}, but "
                f"encoder_out({encoder_output_dim}) + policy({policy_dim}) + commands({commands_dim}) = "
                f"{expected_actor_in}."
            )
        teacher = JumpPolicyAll(
            encoder=encoder,
            actor=actor,
            obs_history_dim=obs_history_dim,
            policy_dim=policy_dim,
            commands_dim=commands_dim,
        )
        input_dim = obs_history_dim + policy_dim + commands_dim
        recommended_obs_groups = ["obsHistory_flat", "policy", "commands"]
    else:
        expected_actor_in = encoder_output_dim + policy_dim + velocity_dim + gait_commands_dim
        if actor_input_dim != expected_actor_in:
            raise ValueError(
                f"Gait actor input mismatch: actor expects {actor_input_dim}, but "
                f"encoder_out({encoder_output_dim}) + policy({policy_dim}) + velocity({velocity_dim}) + "
                f"gait_commands({gait_commands_dim}) = {expected_actor_in}."
            )
        teacher = GaitPolicyAll(
            encoder=encoder,
            actor=actor,
            obs_history_dim=obs_history_dim,
            policy_dim=policy_dim,
            commands_dim=commands_dim,
            gait_commands_dim=gait_commands_dim,
            velocity_dim=velocity_dim,
        )
        input_dim = obs_history_dim + policy_dim + commands_dim + gait_commands_dim
        recommended_obs_groups = ["obsHistory_flat", "policy", "commands", "gait_commands"]

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(args.checkpoint), "exported", "policy_all.pt")
    output = os.path.abspath(os.path.expanduser(output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")

    teacher = teacher.cpu().eval()
    with torch.no_grad():
        dummy = torch.zeros(2, input_dim)
        dummy_out = teacher(dummy)
        if dummy_out.shape[-1] != actor_output_dim:
            raise RuntimeError(
                f"Unexpected output dim: got {dummy_out.shape[-1]}, expected {actor_output_dim}."
            )

    scripted = torch.jit.script(teacher)
    scripted.save(output)

    print("Exported:", output)
    print("Mode:", args.mode)
    print("Input dim:", input_dim)
    print("Output dim:", actor_output_dim)
    print("Recommended obs_groups:", recommended_obs_groups)


if __name__ == "__main__":
    main()

