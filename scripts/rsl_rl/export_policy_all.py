#!/usr/bin/env python3
"""Export fused teacher policy (encoder + actor) as TorchScript.

Builds ``policy_all.pt`` from ONNX files:

1) ``encoder.onnx``
2) ``policy.onnx``

The exported model embeds:

1) encoder(obsHistory_flat)
2) actor input assembly
3) actor forward

Input contract for exported ``policy_all.pt``:

/home/hugh/anaconda3/envs/env_isaaclab_2/bin/python scripts/rsl_rl/export_policy_all.py --encoder-onnx /home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/encoder.onnx --policy-onnx /home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/policy.onnx --output /home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/policy_all.pt --overwrite
Exported: /home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/policy_all.pt
Source: encoder=/home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/encoder.onnx, policy=/home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/policy.onnx
Input dim: 316
Output dim: 8

``x = [obsHistory_flat, actor_tail]``, where
``actor_tail_dim = policy_input_dim - encoder_output_dim``.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn


def _activation_from_onnx_node(node) -> nn.Module | None:
    if node.op_type == "Elu":
        return nn.ELU()
    if node.op_type == "Relu":
        return nn.ReLU()
    if node.op_type == "Tanh":
        return nn.Tanh()
    if node.op_type == "LeakyRelu":
        alpha = 0.01
        for attr in node.attribute:
            if attr.name == "alpha":
                alpha = float(attr.f)
                break
        return nn.LeakyReLU(negative_slope=alpha)
    return None


def _build_mlp_from_onnx(onnx_path: str) -> nn.Sequential:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise ImportError("Loading ONNX files requires the 'onnx' package.") from exc

    onnx_model = onnx.load(onnx_path)
    init_tensors: dict[str, torch.Tensor] = {}
    for initializer in onnx_model.graph.initializer:
        init_tensors[initializer.name] = torch.from_numpy(
            numpy_helper.to_array(initializer).copy()
        ).to(dtype=torch.float32)

    if len(init_tensors) == 0:
        raise ValueError(f"No initializers found in ONNX model: {onnx_path}")

    nodes = list(onnx_model.graph.node)
    modules: list[nn.Module] = []
    node_id = 0

    while node_id < len(nodes):
        node = nodes[node_id]

        if node.op_type == "Identity":
            node_id += 1
            continue

        if node.op_type == "MatMul":
            weight_name = next((name for name in node.input if name in init_tensors), None)
            if weight_name is None:
                raise ValueError(
                    f"MatMul node '{node.name}' has no initializer weight in {onnx_path}."
                )
            raw_weight = init_tensors[weight_name]
            if raw_weight.ndim != 2:
                raise ValueError(
                    f"Expected 2-D MatMul weight, got shape {tuple(raw_weight.shape)} in {onnx_path}."
                )

            node_id += 1
            if node_id >= len(nodes) or nodes[node_id].op_type != "Add":
                raise ValueError(
                    f"Expected Add node after MatMul in {onnx_path}, got "
                    f"'{nodes[node_id].op_type if node_id < len(nodes) else 'EOF'}'."
                )

            add_node = nodes[node_id]
            bias_name = next((name for name in add_node.input if name in init_tensors), None)
            if bias_name is None:
                raise ValueError(f"Add node '{add_node.name}' has no initializer bias in {onnx_path}.")
            bias = init_tensors[bias_name].reshape(-1)

            in_features = int(raw_weight.shape[0])
            out_features = int(raw_weight.shape[1])
            if bias.numel() != out_features:
                raise ValueError(
                    f"Bias size mismatch in {onnx_path}: bias has {bias.numel()} values, "
                    f"expected {out_features}."
                )

            layer = nn.Linear(in_features, out_features, bias=True)
            with torch.no_grad():
                layer.weight.copy_(raw_weight.t().contiguous())
                layer.bias.copy_(bias)
            modules.append(layer)

            node_id += 1
            if node_id < len(nodes):
                activation = _activation_from_onnx_node(nodes[node_id])
                if activation is not None:
                    modules.append(activation)
                    node_id += 1
            continue

        if node.op_type == "Gemm":
            if len(node.input) < 2:
                raise ValueError(f"Gemm node '{node.name}' is missing weight input in {onnx_path}.")

            weight_name = node.input[1]
            if weight_name not in init_tensors:
                raise ValueError(f"Gemm weight '{weight_name}' not found in {onnx_path}.")

            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            alpha = float(attrs.get("alpha", 1.0))
            beta = float(attrs.get("beta", 1.0))
            trans_a = int(attrs.get("transA", 0))
            trans_b = int(attrs.get("transB", 0))

            if trans_a != 0:
                raise ValueError(f"Unsupported Gemm attribute transA={trans_a} in {onnx_path}.")
            if alpha != 1.0 or beta != 1.0:
                raise ValueError(
                    f"Unsupported Gemm scaling alpha={alpha}, beta={beta} in {onnx_path}."
                )

            raw_weight = init_tensors[weight_name]
            if raw_weight.ndim != 2:
                raise ValueError(
                    f"Expected 2-D Gemm weight, got shape {tuple(raw_weight.shape)} in {onnx_path}."
                )

            if trans_b == 0:
                in_features = int(raw_weight.shape[0])
                out_features = int(raw_weight.shape[1])
                linear_weight = raw_weight.t().contiguous()
            elif trans_b == 1:
                out_features = int(raw_weight.shape[0])
                in_features = int(raw_weight.shape[1])
                linear_weight = raw_weight.contiguous()
            else:
                raise ValueError(f"Unsupported Gemm attribute transB={trans_b} in {onnx_path}.")

            bias = torch.zeros(out_features, dtype=torch.float32)
            if len(node.input) >= 3 and node.input[2] in init_tensors:
                bias = init_tensors[node.input[2]].reshape(-1).to(dtype=torch.float32)
            if bias.numel() != out_features:
                raise ValueError(
                    f"Bias size mismatch in {onnx_path}: bias has {bias.numel()} values, "
                    f"expected {out_features}."
                )

            layer = nn.Linear(in_features, out_features, bias=True)
            with torch.no_grad():
                layer.weight.copy_(linear_weight)
                layer.bias.copy_(bias)
            modules.append(layer)

            node_id += 1
            if node_id < len(nodes):
                activation = _activation_from_onnx_node(nodes[node_id])
                if activation is not None:
                    modules.append(activation)
                    node_id += 1
            continue

        raise ValueError(
            f"Unsupported ONNX node '{node.op_type}' while parsing {onnx_path}. "
            "Expected an MLP-like sequence of Linear(+bias)+activation blocks."
        )

    if len(modules) == 0 or not any(isinstance(module, nn.Linear) for module in modules):
        raise ValueError(f"Failed to parse any Linear layers from ONNX model: {onnx_path}")

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


class PolicyAll(nn.Module):
    """Fused policy: encoder(obsHistory_flat) + remaining tail -> actor."""

    def __init__(
        self,
        encoder: nn.Sequential,
        actor: nn.Sequential,
        obs_history_dim: int,
        actor_input_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.obs_history_dim = obs_history_dim
        self.actor_input_dim = actor_input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs_history = x[..., : self.obs_history_dim]
        tail = x[..., self.obs_history_dim :]

        encoder_out = self.encoder(obs_history)
        actor_in = torch.cat((encoder_out, tail), dim=-1)
        actor_in_dim = int(actor_in.shape[-1])
        if actor_in_dim != self.actor_input_dim:
            raise RuntimeError(
                f"Input dim mismatch: actor_in={actor_in_dim}, actor_expected={self.actor_input_dim}."
            )
        return self.actor(actor_in)

    @torch.jit.export
    def reset(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export fused policy_all.pt from encoder.onnx + policy.onnx."
    )
    parser.add_argument("--encoder-onnx", type=str, required=True, help="Path to encoder.onnx.")
    parser.add_argument("--policy-onnx", type=str, required=True, help="Path to policy.onnx.")
    parser.add_argument("--output", type=str, default=None, help="Output path for policy_all.pt.")
    parser.add_argument(
        "--obs-history-dim",
        type=int,
        default=280,
        help="Override obsHistory_flat dim (auto from encoder input if unset).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    args = parser.parse_args()

    encoder_onnx = os.path.abspath(os.path.expanduser(args.encoder_onnx))
    policy_onnx = os.path.abspath(os.path.expanduser(args.policy_onnx))
    if not os.path.isfile(encoder_onnx):
        raise FileNotFoundError(f"encoder.onnx not found: {encoder_onnx}")
    if not os.path.isfile(policy_onnx):
        raise FileNotFoundError(f"policy.onnx not found: {policy_onnx}")

    encoder = _build_mlp_from_onnx(encoder_onnx).eval()
    actor = _build_mlp_from_onnx(policy_onnx).eval()

    actor_input_dim = _first_linear_in_dim(actor)
    actor_output_dim = _last_linear_out_dim(actor)
    encoder_input_dim = _first_linear_in_dim(encoder)
    encoder_output_dim = _last_linear_out_dim(encoder)

    obs_history_dim = encoder_input_dim if args.obs_history_dim is None else args.obs_history_dim
    if obs_history_dim != encoder_input_dim:
        raise ValueError(
            f"obs_history_dim mismatch: provided {obs_history_dim}, encoder expects {encoder_input_dim}."
        )

    actor_tail_dim = actor_input_dim - encoder_output_dim
    if actor_tail_dim <= 0:
        raise ValueError(
            f"Invalid dims: actor_input_dim({actor_input_dim}) - encoder_output_dim({encoder_output_dim}) "
            f"= {actor_tail_dim}."
        )

    teacher = PolicyAll(
        encoder=encoder,
        actor=actor,
        obs_history_dim=obs_history_dim,
        actor_input_dim=actor_input_dim,
    ).cpu().eval()

    output = args.output
    if output is None:
        output = os.path.join(os.path.dirname(policy_onnx), "policy_all.pt")
    output = os.path.abspath(os.path.expanduser(output))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if os.path.exists(output) and not args.overwrite:
        raise FileExistsError(f"Output exists: {output}. Use --overwrite to replace it.")

    direct_input_dim = obs_history_dim + actor_tail_dim
    with torch.no_grad():
        dummy = torch.zeros(2, direct_input_dim)
        dummy_out = teacher(dummy)
        if dummy_out.shape[-1] != actor_output_dim:
            raise RuntimeError(
                f"Unexpected output dim: got {dummy_out.shape[-1]}, expected {actor_output_dim}."
            )

    scripted = torch.jit.script(teacher)
    scripted.save(output)

    print("Exported:", output)
    print("Source:", f"encoder={encoder_onnx}, policy={policy_onnx}")
    print("Input dim:", direct_input_dim)
    print("Output dim:", actor_output_dim)


if __name__ == "__main__":
    main()
