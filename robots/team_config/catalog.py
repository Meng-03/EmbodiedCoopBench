from __future__ import annotations

from copy import deepcopy
from typing import Any


def normalize_action_names(action_specs_or_names: list[Any]) -> list[str]:
    """Normalize tool/action definitions to a deduplicated action-name list."""
    names: list[str] = []
    seen: set[str] = set()
    for item in action_specs_or_names:
        if isinstance(item, dict):
            raw = item.get("name")
        else:
            raw = item
        name = str(raw or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def build_default_two_robot_presets(action_names: list[str]) -> dict[str, dict[str, Any]]:
    """Build common 2-robot presets from a semantic action name list."""
    all_actions = list(action_names)
    all_actions_without_clean = [a for a in all_actions if a != "clean"]
    return {
        "homogeneous_full_2": {
            "description": "2 robots, same full action set",
            "robot_ids": [0, 1],
            "robot_action_sets": None,
        },
        "heterogeneous_robot1_no_clean_2": {
            "description": "2 robots, robot_0 full set, robot_1 without clean",
            "robot_ids": [0, 1],
            "robot_action_sets": {
                0: list(all_actions),
                1: list(all_actions_without_clean),
            },
        },
        "heterogeneous_robot2_no_clean_3": {
            "description": "3 robots, robot_0 and robot_1 full set, robot_2 without clean",
            "robot_ids": [0, 1, 2],
            "robot_action_sets": {
                0: list(all_actions),
                1: list(all_actions),
                2: list(all_actions_without_clean),
            },
        },
    }


class RobotTeamConfigCatalog:
    """Reusable robot-team config catalog for different agents."""
    def __init__(
        self,
        action_specs_or_names: list[Any],
        presets: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.action_names = normalize_action_names(action_specs_or_names)
        if not self.action_names:
            raise ValueError("RobotTeamConfigCatalog requires at least one action name")

        self.presets: dict[str, dict[str, Any]] = (
            deepcopy(presets)
            if presets is not None
            else build_default_two_robot_presets(self.action_names)
        )

    def list_presets(self) -> list[str]:
        return sorted(self.presets.keys())

    def get_config(self, preset_name: str) -> dict[str, Any]:
        if preset_name not in self.presets:
            available = ", ".join(self.list_presets())
            raise ValueError(f"Unknown robot team preset: {preset_name}. Available: {available}")

        config = deepcopy(self.presets[preset_name])
        robot_ids = [int(rid) for rid in config.get("robot_ids", [])]
        if not robot_ids:
            raise ValueError(f"Invalid preset {preset_name}: robot_ids cannot be empty")

        action_sets = config.get("robot_action_sets")
        if action_sets is not None:
            valid_actions = set(self.action_names)
            normalized: dict[int, list[str]] = {}
            for rid, actions in action_sets.items():
                rid_int = int(rid)
                if rid_int not in robot_ids:
                    raise ValueError(
                        f"Invalid preset {preset_name}: robot_action_sets has robot {rid_int} not in robot_ids"
                    )

                checked: list[str] = []
                for action in actions or []:
                    action_name = str(action).strip()
                    if action_name not in valid_actions:
                        raise ValueError(
                            f"Invalid preset {preset_name}: unsupported action '{action_name}' for robot {rid_int}"
                        )
                    checked.append(action_name)
                normalized[rid_int] = checked
            config["robot_action_sets"] = normalized

        config["robot_ids"] = robot_ids
        return config
