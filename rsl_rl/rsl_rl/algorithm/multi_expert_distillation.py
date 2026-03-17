# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.modules import MLPModel
from rsl_rl.storage import DistillationRolloutStorage as RolloutStorage
from rsl_rl.utils import (
    expand_obs_group_mapping,
    expand_obs_groups,
    resolve_callable,
    resolve_obs_groups,
    resolve_optimizer,
)


class JITTeacherWrapper(nn.Module):
    """Wraps a TorchScript (JIT) exported policy for use as a frozen expert teacher.

    Accepts TensorDict observations, concatenates the configured observation groups into a flat
    tensor, and forwards through the JIT model. Allows JIT-exported ``policy.pt`` files to be
    used as teachers without re-specifying the model architecture.
    """

    is_recurrent: bool = False

    def __init__(self, jit_model: torch.jit.ScriptModule, obs_groups: list[str]) -> None:
        super().__init__()
        self.jit_model = jit_model
        self._obs_groups = obs_groups

    def forward(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        latent = torch.cat([obs[g] for g in self._obs_groups], dim=-1)
        return self.jit_model(latent)

    def reset(self, dones: torch.Tensor | None = None, **kwargs) -> None:
        if hasattr(self.jit_model, "reset"):
            self.jit_model.reset()

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass


class MultiExpertDistillation:
    """Distillation algorithm for training one student against multiple expert teachers.

    Each environment is assigned to one expert via an observation group (``teacher_id_obs_group``).
    The student learns to imitate the assigned expert through behavior cloning.

    Teachers are loaded from JIT (TorchScript) checkpoints at construction time via
    :meth:`construct_algorithm`. The student is an :class:`~rsl_rl.modules.MLPModel`.

    Example config structure::

        cfg = {
            "student": {
                "class_name": "rsl_rl.modules.MLPModel",
                "hidden_dims": [512, 256, 128],
                "obs_normalization": False,
                "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0},
            },
            "algorithm": {
                "class_name": "MultiExpertDistillation",
                "num_learning_epochs": 1,
                "gradient_length": 15,
                "learning_rate": 1e-3,
                "loss_type": "mse",
            },
            "obs_groups": {
                "student": ["policy"],  # student obs set
            },
            "experts": [
                {
                    "name": "flat_expert",
                    "obs_groups": ["policy", "critic"],
                    "jit_policy_path": "/path/to/flat_expert.pt",
                },
                {
                    "name": "stairs_expert",
                    "obs_groups": ["policy", "critic"],
                    "jit_policy_path": "/path/to/stairs_expert.pt",
                },
            ],
            "teacher_id_obs_group": "env_group",  # obs key that indicates expert id per env
            "num_steps_per_env": 24,
            "multi_gpu": None,
        }
    """

    student: MLPModel
    teachers: nn.ModuleList
    teacher_loaded: bool = False

    def __init__(
        self,
        student: MLPModel,
        teachers: list[nn.Module],
        storage: RolloutStorage,
        env: VecEnv,
        expert_names: list[str],
        teacher_id_obs_group: str = "env_group",
        num_learning_epochs: int = 1,
        gradient_length: int = 15,
        learning_rate: float = 1e-3,
        max_grad_norm: float | None = None,
        loss_type: str = "mse",
        optimizer: str = "adam",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ) -> None:
        del kwargs
        self.device = device
        self.env = env
        self.teacher_id_obs_group = teacher_id_obs_group
        self.expert_names = expert_names
        self.is_multi_gpu = multi_gpu_cfg is not None

        if len(teachers) == 0:
            raise ValueError("MultiExpertDistillation requires at least one teacher.")
        if len(teachers) != len(expert_names):
            raise ValueError("Number of teacher models must match number of expert names.")
        if len(set(expert_names)) != len(expert_names):
            raise ValueError(f"Expert names must be unique. Received: {expert_names}")

        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.student = student.to(self.device)
        self.teachers = nn.ModuleList([teacher.to(self.device) for teacher in teachers])
        self.optimizer = resolve_optimizer(optimizer)(self.student.parameters(), lr=learning_rate)

        self.storage = storage
        self.transition = RolloutStorage.Transition()
        self.last_hidden_state = None

        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type not in loss_fn_dict:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported: {list(loss_fn_dict.keys())}")
        self.loss_fn = loss_fn_dict[loss_type]

        self.num_updates = 0
        self.teacher_loaded = True

    def _decode_teacher_ids(self, teacher_id_obs: torch.Tensor) -> torch.Tensor:
        teacher_id_obs = teacher_id_obs.to(device=self.device)
        if teacher_id_obs.ndim == 1:
            teacher_ids = teacher_id_obs.to(dtype=torch.long)
        elif teacher_id_obs.shape[-1] == 1:
            teacher_ids = torch.round(teacher_id_obs.squeeze(-1)).to(dtype=torch.long)
        else:
            teacher_ids = torch.argmax(teacher_id_obs, dim=-1).to(dtype=torch.long)
        return teacher_ids.view(-1)

    def _resolve_teacher_ids(self, obs: TensorDict) -> torch.Tensor:
        if self.teacher_id_obs_group not in obs:
            raise KeyError(
                f"Observation group '{self.teacher_id_obs_group}' is required for teacher routing. "
                f"Available groups: {list(obs.keys())}"
            )
        teacher_ids = self._decode_teacher_ids(obs[self.teacher_id_obs_group])
        if teacher_ids.shape[0] != self.env.num_envs:
            raise ValueError(
                f"Teacher id buffer must have shape ({self.env.num_envs},), got {tuple(teacher_ids.shape)}."
            )
        if torch.any(teacher_ids < 0) or torch.any(teacher_ids >= len(self.teachers)):
            raise ValueError(
                f"Teacher ids must be within [0, {len(self.teachers) - 1}], "
                f"got {teacher_ids.unique().tolist()}."
            )
        return teacher_ids

    def _compute_per_env_loss(self, actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        losses = self.loss_fn(actions, targets, reduction="none")
        return losses.view(losses.shape[0], -1).mean(dim=1)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample student actions and gather expert actions for the current env grouping."""
        teacher_ids = self._resolve_teacher_ids(obs)

        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        self.transition.privileged_actions = torch.zeros_like(self.transition.actions)
        self.transition.observations = obs

        for teacher_id, teacher in enumerate(self.teachers):
            env_ids = torch.nonzero(teacher_ids == teacher_id, as_tuple=False).squeeze(-1)
            if env_ids.numel() == 0:
                continue
            self.transition.privileged_actions[env_ids] = teacher(obs[env_ids]).detach()

        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        """Record the current transition and update recurrent state."""
        self.student.update_normalization(obs)
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        for teacher in self.teachers:
            teacher.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """No-op — distillation does not use returns."""
        pass

    def update(self) -> dict[str, float]:
        """Run one optimization pass over the rollout buffer."""
        self.num_updates += 1
        mean_behavior_loss = 0.0
        accumulated_loss: torch.Tensor | int = 0
        num_batches = 0

        group_loss_totals = {name: 0.0 for name in self.expert_names}
        group_loss_denominators = {name: 0 for name in self.expert_names}

        for _epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_state)
            self.student.detach_hidden_state()
            for batch in self.storage.batch_generator():
                actions = self.student(batch.observations)
                per_env_loss = self._compute_per_env_loss(actions, batch.privileged_actions)
                behavior_loss = per_env_loss.mean()

                accumulated_loss = accumulated_loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                num_batches += 1

                if self.teacher_id_obs_group not in batch.observations:
                    raise KeyError(
                        f"Observation group '{self.teacher_id_obs_group}' required for per-expert metrics. "
                        f"Available groups: {list(batch.observations.keys())}"
                    )
                teacher_ids = self._decode_teacher_ids(batch.observations[self.teacher_id_obs_group])
                for teacher_id, expert_name in enumerate(self.expert_names):
                    mask = teacher_ids == teacher_id
                    if torch.any(mask):
                        group_loss_totals[expert_name] += per_env_loss[mask].sum().item()
                        group_loss_denominators[expert_name] += int(mask.sum().item())

                if num_batches % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    accumulated_loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    accumulated_loss = 0

                done_mask = batch.dones.view(-1) if batch.dones is not None else None
                self.student.reset(done_mask)
                self.student.detach_hidden_state(done_mask)

        mean_behavior_loss /= max(num_batches, 1)
        self.storage.clear()
        self.last_hidden_state = self.student.get_hidden_state()
        self.student.detach_hidden_state()

        loss_dict: dict[str, float] = {"behavior": mean_behavior_loss}
        for expert_name in self.expert_names:
            denominator = group_loss_denominators[expert_name]
            loss_dict[f"behavior_{expert_name}"] = (
                group_loss_totals[expert_name] / denominator if denominator > 0 else 0.0
            )
        return loss_dict

    def train_mode(self) -> None:
        """Set student to train mode, teachers remain frozen in eval mode."""
        self.student.train()
        for teacher in self.teachers:
            teacher.eval()

    def eval_mode(self) -> None:
        self.student.eval()
        for teacher in self.teachers:
            teacher.eval()

    def save(self) -> dict:
        """Return a serializable state dict for checkpointing."""
        return {
            "student_state_dict": self.student.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "expert_names": self.expert_names,
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load a checkpoint. Returns whether to restore the training iteration counter."""
        if load_cfg is None and any("actor_state_dict" in key for key in loaded_dict):
            load_cfg = {"student": True, "optimizer": False, "iteration": False}
        elif load_cfg is None:
            load_cfg = {"student": True, "optimizer": True, "iteration": True}

        if load_cfg.get("student"):
            student_state = loaded_dict.get("student_state_dict") or loaded_dict.get("actor_state_dict")
            if student_state is None:
                raise KeyError("Checkpoint does not contain 'student_state_dict' or 'actor_state_dict'.")
            self.student.load_state_dict(student_state, strict=strict)
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Return the trainable student policy."""
        return self.student

    def broadcast_parameters(self) -> None:
        """Broadcast student parameters from rank 0 across workers."""
        model_params = [self.student.state_dict(), [teacher.state_dict() for teacher in self.teachers]]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.student.load_state_dict(model_params[0])
        for teacher, teacher_state in zip(self.teachers, model_params[1], strict=True):
            teacher.load_state_dict(teacher_state)

    def reduce_parameters(self) -> None:
        """Average student gradients across distributed workers."""
        grads = [param.grad.view(-1) for param in self.student.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.student.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset: offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def _validate_expert_obs_groups(obs: TensorDict, expert_name: str, obs_groups: list[str]) -> None:
        if len(obs_groups) == 0:
            raise ValueError(f"Expert '{expert_name}' must define at least one observation group.")
        for obs_group in obs_groups:
            if obs_group not in obs:
                raise ValueError(
                    f"Observation '{obs_group}' for expert '{expert_name}' not found. "
                    f"Available groups: {list(obs.keys())}"
                )

    @classmethod
    def _build_teacher(
        cls,
        obs: TensorDict,
        env: VecEnv,
        expert_cfg: dict,
        device: str,
    ) -> tuple[str, nn.Module]:
        expert_name = expert_cfg["name"]
        obs_groups = expand_obs_groups(list(expert_cfg["obs_groups"]))
        cls._validate_expert_obs_groups(obs, expert_name, obs_groups)

        jit_policy_path = expert_cfg["jit_policy_path"]
        jit_model = torch.jit.load(jit_policy_path, map_location=device)
        jit_model.eval()
        teacher = JITTeacherWrapper(jit_model, obs_groups)
        print(f"Teacher [{expert_name}]: loaded from JIT {jit_policy_path}")
        return expert_name, teacher

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> "MultiExpertDistillation":
        """Build the distillation algorithm, student, teachers, and storage from config.

        Args:
            obs: Initial observations from the environment (TensorDict).
            env: The vectorized environment.
            cfg: Configuration dictionary (see class docstring for structure).
            device: Compute device string (e.g. ``"cuda:0"``).

        Returns:
            Fully initialized :class:`MultiExpertDistillation` instance.
        """
        alg_class: type[MultiExpertDistillation] = resolve_callable(cfg["algorithm"].pop("class_name"))
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))

        default_sets = ["student"]
        cfg["obs_groups"] = expand_obs_group_mapping(dict(cfg["obs_groups"]))
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        student: MLPModel = student_class(
            obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]
        ).to(device)
        print(f"Student Model: {student}")

        expert_names: list[str] = []
        teachers: list[nn.Module] = []
        for expert_cfg in cfg["experts"]:
            expert_name, teacher = alg_class._build_teacher(obs, env, expert_cfg, device)
            expert_names.append(expert_name)
            teachers.append(teacher)

        storage = RolloutStorage(
            "distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )
        alg = alg_class(
            student=student,
            teachers=teachers,
            storage=storage,
            env=env,
            expert_names=expert_names,
            teacher_id_obs_group=cfg.get("teacher_id_obs_group", "env_group"),
            device=device,
            multi_gpu_cfg=cfg.get("multi_gpu"),
            **cfg["algorithm"],
        )
        return alg
