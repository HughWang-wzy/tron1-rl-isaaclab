"""Play an Isaac Lab task using a TorchScript policy directly.

This script is intended for validating exported JIT models (for example
``policy_all.pt``) without loading an RSL-RL runner checkpoint.


python scripts/rsl_rl/play_jit.py \
  --task Isaac-Limx-WF-Jump-Rough-Play-v0 \
  --jit_policy_path /home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_jump/2026-03-17_23-42-13-rough/exported/policy_all.pt \
  --num_envs 32
  
python scripts/rsl_rl/play_jit.py \
  --task Isaac-Limx-WF-Gait-Flat-Play-v0 \ 
  --jit_policy_path /home/hugh/tron1-rl-isaaclab/logs/rsl_rl/wf_tron_1a_gait/2026-03-15_19-02-27/exported/policy_all.pt \      
  --obs_groups obsHistory,policy,commands \
  --num_envs 16 \
  --device cuda:0 \
  --max_steps 10000 \
  --print_every 200
"""

from __future__ import annotations

import argparse
import itertools
import os
from collections.abc import Sequence

import torch
from isaaclab.app import AppLauncher


def _read_positive_int_attr(module: torch.jit.ScriptModule, attr_name: str) -> int | None:
    """Read positive integer TorchScript attributes when available."""
    if not hasattr(module, attr_name):
        return None
    raw_value = getattr(module, attr_name)
    if isinstance(raw_value, torch.Tensor):
        if raw_value.numel() != 1:
            return None
        value = int(raw_value.item())
    elif isinstance(raw_value, (int, float)):
        value = int(raw_value)
    else:
        return None
    return value if value > 0 else None


def _infer_jit_layout(jit_model: torch.jit.ScriptModule) -> dict[str, int]:
    """Infer layout metadata embedded in exported JIT models."""
    layout: dict[str, int] = {}
    for key in (
        "obs_history_dim",
        "policy_dim",
        "commands_dim",
        "gait_commands_dim",
        "jump_commands_dim",
    ):
        value = _read_positive_int_attr(jit_model, key)
        if value is not None:
            layout[key] = value
    return layout


def _infer_expected_input_dim(jit_model: torch.jit.ScriptModule, layout: dict[str, int]) -> int | None:
    """Infer the model's expected input dim from explicit JIT metadata."""
    for key in ("input_dim", "obs_dim", "input_size"):
        value = _read_positive_int_attr(jit_model, key)
        if value is not None:
            return value

    if len(layout) == 0:
        return None

    # Some exported policies only carry obs_history_dim even when real model input
    # also includes policy/commands groups. Treat this as partial metadata.
    if layout.keys() == {"obs_history_dim"}:
        return None

    input_dim = 0
    input_dim += layout.get("obs_history_dim", 0)
    input_dim += layout.get("policy_dim", 0)
    input_dim += layout.get("commands_dim", 0)
    input_dim += layout.get("gait_commands_dim", 0)
    input_dim += layout.get("jump_commands_dim", 0)
    return input_dim if input_dim > 0 else None


def _obs_group_tensor(obs: dict | torch.Tensor, group_name: str) -> torch.Tensor:
    """Fetch one observation group and flatten history-style 3D tensors."""
    tensor = obs[group_name]
    if tensor.ndim == 3:
        return tensor.flatten(start_dim=1)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"Observation group '{group_name}' must be 2D/3D, got shape {tuple(tensor.shape)}.")


def _resolve_obs_group_dims(obs: dict | torch.Tensor, obs_groups: Sequence[str]) -> tuple[dict[str, int], int]:
    """Resolve per-group and total dims after flattening history tensors."""
    group_dims: dict[str, int] = {}
    total_dim = 0
    for group_name in obs_groups:
        group_tensor = _obs_group_tensor(obs, group_name)
        dim = int(group_tensor.shape[-1])
        group_dims[group_name] = dim
        total_dim += dim
    return group_dims, total_dim


def _infer_obs_groups_auto(obs: dict | torch.Tensor, layout: dict[str, int], expected_dim: int | None) -> list[str]:
    """Infer observation-group order for JIT inference."""
    available = set(obs.keys())

    ordered_from_layout: list[str] = []
    mapping = [
        ("obs_history_dim", ("obsHistory_flat", "obsHistory")),
        ("policy_dim", ("policy",)),
        ("commands_dim", ("commands",)),
        ("gait_commands_dim", ("gait_commands",)),
        ("jump_commands_dim", ("jump_commands",)),
    ]
    for layout_key, candidates in mapping:
        if layout.get(layout_key, 0) <= 0:
            continue
        picked = next((candidate for candidate in candidates if candidate in available), None)
        if picked is None:
            raise ValueError(
                f"JIT expects '{layout_key}' but none of candidate groups {list(candidates)} exists in env obs. "
                f"Available groups: {sorted(available)}"
            )
        ordered_from_layout.append(picked)

    if len(ordered_from_layout) > 0:
        return ordered_from_layout

    # Fallback for models without explicit layout metadata.
    fallback_order = [
        "obsHistory_flat",
        "obsHistory",
        "policy",
        "commands",
        "gait_commands",
        "jump_commands",
        "env_group",
    ]
    candidates = [group_name for group_name in fallback_order if group_name in available]
    if len(candidates) == 0:
        raise ValueError(f"No known observation groups found. Available groups: {sorted(available)}")

    if expected_dim is None:
        if "policy" in candidates:
            return ["policy"]
        return [candidates[0]]

    matched: list[list[str]] = []
    for count in range(1, len(candidates) + 1):
        for combo in itertools.combinations(candidates, count):
            combo_list = list(combo)
            _, total_dim = _resolve_obs_group_dims(obs, combo_list)
            if total_dim == expected_dim:
                matched.append(combo_list)

    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise ValueError(
            f"Cannot auto-infer a unique obs_groups combination for expected_dim={expected_dim}. "
            f"Candidate matches: {matched}. Please pass --obs_groups explicitly."
        )

    raise ValueError(
        f"Cannot auto-infer obs_groups for expected_dim={expected_dim}. "
        f"Available groups: {sorted(available)}. Please pass --obs_groups explicitly."
    )


def _parse_obs_groups_arg(obs_groups_arg: str) -> list[str] | None:
    if obs_groups_arg.strip().lower() == "auto":
        return None
    parsed = [item.strip() for item in obs_groups_arg.split(",") if item.strip()]
    if len(parsed) == 0:
        raise ValueError("Empty --obs_groups received. Use comma-separated names or 'auto'.")
    return parsed


parser = argparse.ArgumentParser(description="Play an Isaac Lab task with a TorchScript policy directly.")
parser.add_argument("--task", type=str, required=True, help="Environment task name.")
parser.add_argument("--jit_policy_path", type=str, required=True, help="Path to exported TorchScript policy (.pt).")
parser.add_argument(
    "--obs_groups",
    type=str,
    default="auto",
    help=(
        "Comma-separated obs groups in concatenation order (e.g. "
        "'obsHistory_flat,policy,commands'). Use 'auto' to infer from JIT metadata."
    ),
)
parser.add_argument("--video", action="store_true", default=False, help="Record a play video.")
parser.add_argument("--video_length", type=int, default=400, help="Recorded video length in steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric/USD optimizations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--max_steps", type=int, default=0, help="Maximum sim steps; 0 means run until app closes.")
parser.add_argument("--print_every", type=int, default=200, help="Print runtime stats every N steps.")
parser.add_argument(
    "--clip_actions",
    type=float,
    default=None,
    help="Optional action clipping range. If set, actions are clamped to [-clip_actions, clip_actions].",
)
print("[INFO] Parsing command-line arguments...", flush=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Register custom tasks.
import bipedal_locomotion  # noqa: F401


def main() -> None:
    print("[DEBUG] Parsing env cfg...", flush=True)
    env_cfg: ManagerBasedRLEnvCfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(os.path.dirname(os.path.abspath(args_cli.jit_policy_path)), "videos", "jit"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video:", flush=True)
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    print("[DEBUG] Wrapping env with RslRlVecEnvWrapper...", flush=True)
    env = RslRlVecEnvWrapper(env)
    print("[DEBUG] RslRlVecEnvWrapper done.", flush=True)
    jit_path = os.path.abspath(os.path.expanduser(args_cli.jit_policy_path))
    if not os.path.isfile(jit_path):
        raise FileNotFoundError(f"JIT policy not found: {jit_path}")
    policy_device = env.unwrapped.device
    print(f"[DEBUG] Loading JIT model from {jit_path} to {policy_device}...", flush=True)
    jit_model = torch.jit.load(jit_path, map_location="cpu")
    print("[DEBUG] JIT loaded to CPU, moving to device...", flush=True)
    jit_model = jit_model.to(policy_device)
    jit_model.eval()
    print("[DEBUG] JIT model ready.", flush=True)

    print("[DEBUG] Calling env.reset() to fetch initial observations...", flush=True)
    obs, _ = env.reset()
    print("[DEBUG] env.reset() done.", flush=True)
    print("[DEBUG] Inferring JIT layout...", flush=True)
    layout = _infer_jit_layout(jit_model)
    print(f"[DEBUG] JIT layout inferred: {layout}", flush=True)
    print("[DEBUG] Inferring expected input dim...", flush=True)
    expected_input_dim = _infer_expected_input_dim(jit_model, layout)
    print(f"[DEBUG] expected_input_dim={expected_input_dim}", flush=True)

    print("[DEBUG] Parsing obs_groups arg...", flush=True)
    parsed_obs_groups = _parse_obs_groups_arg(args_cli.obs_groups)
    if parsed_obs_groups is None:
        obs_groups = _infer_obs_groups_auto(obs, layout, expected_input_dim)
    else:
        obs_groups = parsed_obs_groups
    print(f"[DEBUG] obs_groups={obs_groups}", flush=True)

    print("[DEBUG] Validating obs groups existence...", flush=True)
    for group_name in obs_groups:
        if group_name not in obs:
            raise KeyError(
                f"obs group '{group_name}' is not available. Available groups: {sorted(obs.keys())}"
            )
    print("[DEBUG] Obs groups existence check done.", flush=True)

    print("[DEBUG] Resolving obs group dims...", flush=True)
    group_dims, total_input_dim = _resolve_obs_group_dims(obs, obs_groups)
    print(f"[DEBUG] group_dims={group_dims}, total_input_dim={total_input_dim}", flush=True)
    if expected_input_dim is not None and total_input_dim != expected_input_dim:
        raise ValueError(
            f"Input dim mismatch: obs_groups={obs_groups}, group_dims={group_dims}, "
            f"total={total_input_dim}, expected={expected_input_dim}."
        )

    print("[DEBUG] Running preflight forward...", flush=True)
    with torch.inference_mode():
        model_input = torch.cat([_obs_group_tensor(obs, group_name) for group_name in obs_groups], dim=-1)
        model_input = model_input.to(policy_device)
        model_output = jit_model(model_input)
        if model_output.ndim != 2:
            raise RuntimeError(f"JIT output must be 2D [N, A], got {tuple(model_output.shape)}.")
        if int(model_output.shape[-1]) != int(env.num_actions):
            raise RuntimeError(
                f"JIT output action dim mismatch: got {int(model_output.shape[-1])}, expected {int(env.num_actions)}."
            )
    print("[DEBUG] Preflight forward done.", flush=True)

    print(f"[INFO] Loaded JIT: {jit_path}", flush=True)
    print(f"[INFO] Available obs groups: {sorted(obs.keys())}", flush=True)
    print(
        f"[INFO] Using obs_groups={obs_groups} | group_dims={group_dims} | "
        f"input_dim={total_input_dim} | expected_input_dim={expected_input_dim} | layout={layout}"
        ,
        flush=True,
    )
    print(f"[INFO] Action dim: {int(env.num_actions)}", flush=True)
    print("[INFO] Entering simulation loop...", flush=True)

    step = 0
    print("[DEBUG] Initial obs keys and shapes:", flush=True)
    for key, value in obs.items():
        print(f"[DEBUG]  {key}: {tuple(value.shape)}", flush=True)
    while simulation_app.is_running():
        with torch.inference_mode():
            model_input = torch.cat([_obs_group_tensor(obs, group_name) for group_name in obs_groups], dim=-1)
            model_input = model_input.to(policy_device)
            actions = jit_model(model_input)
            if args_cli.clip_actions is not None:
                actions = torch.clamp(actions, -args_cli.clip_actions, args_cli.clip_actions)

            if step == 0:
                actions_finite = bool(torch.isfinite(actions).all().item())
                actions_mean_abs = float(actions.abs().mean().item())
                actions_max_abs = float(actions.abs().max().item())
                print(
                    f"[DEBUG] first_action finite={actions_finite} "
                    f"mean_abs={actions_mean_abs:.4f} max_abs={actions_max_abs:.4f}",
                    flush=True,
                )
                if not actions_finite:
                    raise RuntimeError("First action tensor contains NaN/Inf. Please check model export or obs groups.")

            obs, rewards, dones, infos = env.step(actions)

        step += 1
        if args_cli.print_every > 0 and step % args_cli.print_every == 0:
            mean_reward = float(rewards.mean().item())
            dones_bool = dones.bool() if dones.dtype == torch.bool else (dones > 0)
            done_rate = float(dones_bool.float().mean().item())

            timeout_mask: torch.Tensor | None = None
            if isinstance(infos, dict):
                for timeout_key in ("time_outs", "timeouts", "time_out"):
                    if timeout_key not in infos:
                        continue
                    timeout_raw = infos[timeout_key]
                    timeout_tensor = timeout_raw if isinstance(timeout_raw, torch.Tensor) else None
                    if timeout_tensor is None:
                        continue
                    timeout_tensor = timeout_tensor.view(-1)
                    if timeout_tensor.numel() != dones_bool.numel():
                        continue
                    timeout_mask = timeout_tensor.to(device=dones_bool.device).bool()
                    break

            if timeout_mask is None:
                print(
                    f"[PLAY] step={step} mean_reward={mean_reward:.4f} "
                    f"done_rate={done_rate:.4f} timeout_rate=NA fail_rate=NA"
                    ,
                    flush=True,
                )
            else:
                timeout_rate = float(timeout_mask.float().mean().item())
                fail_rate = float((dones_bool & (~timeout_mask)).float().mean().item())
                print(
                    f"[PLAY] step={step} mean_reward={mean_reward:.4f} "
                    f"done_rate={done_rate:.4f} timeout_rate={timeout_rate:.4f} fail_rate={fail_rate:.4f}"
                    ,
                    flush=True,
                )

        if args_cli.max_steps > 0 and step >= args_cli.max_steps:
            print(f"[INFO] Reached max_steps={args_cli.max_steps}.", flush=True)
            break

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
