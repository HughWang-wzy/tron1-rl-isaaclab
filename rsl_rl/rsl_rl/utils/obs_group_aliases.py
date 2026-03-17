# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility helpers for multi-distillation observation groups.

This module provides legacy compatibility for older teacher checkpoints that use
the deprecated observation group layout:  ["proprio_pre", "command", "proprio_post"]
instead of the canonical:  ["proprioception", "command"]

Once legacy artifacts are retired, this module can be removed.
"""

from __future__ import annotations

import sys

_RED = "\033[31m"
_RESET = "\033[0m"

PROPRIOCEPTION_GROUP = "proprioception"
COMMAND_GROUP = "command"
DEPRECATED_GROUPS = frozenset({"proprio_pre", "proprio_post"})
_LEGACY_COMPAT_SEQUENCE = ("proprio_pre", "command", "proprio_post")


def _warn_deprecated_obs_groups(obs_groups: list[str]) -> None:
    deprecated = [group for group in obs_groups if group in DEPRECATED_GROUPS]
    if not deprecated:
        return
    message = (
        f"{_RED}[multi_distill] Deprecated observation groups detected: {deprecated}. "
        f"Use ['{PROPRIOCEPTION_GROUP}', '{COMMAND_GROUP}'] instead.{_RESET}\n"
    )
    sys.stderr.write(message)
    sys.stderr.flush()


def expand_obs_groups(obs_groups: list[str]) -> list[str]:
    """Expand canonical multi-distillation groups into the legacy checkpoint layout."""
    resolved_groups = list(obs_groups)
    _warn_deprecated_obs_groups(resolved_groups)

    if PROPRIOCEPTION_GROUP in resolved_groups and any(group in DEPRECATED_GROUPS for group in resolved_groups):
        raise ValueError(
            "Do not mix deprecated observation groups ['proprio_pre', 'proprio_post'] with "
            "the canonical 'proprioception' group."
        )

    if any(group in DEPRECATED_GROUPS for group in resolved_groups):
        return resolved_groups

    if resolved_groups == [PROPRIOCEPTION_GROUP, COMMAND_GROUP]:
        return list(_LEGACY_COMPAT_SEQUENCE)

    expanded_groups: list[str] = []
    idx = 0
    while idx < len(resolved_groups):
        group_name = resolved_groups[idx]
        if (
            group_name == PROPRIOCEPTION_GROUP
            and idx + 1 < len(resolved_groups)
            and resolved_groups[idx + 1] == COMMAND_GROUP
        ):
            expanded_groups.extend(_LEGACY_COMPAT_SEQUENCE)
            idx += 2
            continue
        if group_name == PROPRIOCEPTION_GROUP:
            expanded_groups.extend(["proprio_pre", "proprio_post"])
        else:
            expanded_groups.append(group_name)
        idx += 1
    return expanded_groups


def expand_obs_group_mapping(obs_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    """Expand every observation-set entry through the legacy compatibility mapping."""
    return {obs_set: expand_obs_groups(groups) for obs_set, groups in obs_groups.items()}
