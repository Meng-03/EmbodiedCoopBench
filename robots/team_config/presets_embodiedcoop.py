from __future__ import annotations

from typing import Any

from agents.embodiedcoop.toolregistry import (
    FACILITY_INSPECTION_ACTIONS,
    RESTAURANT_SERVING_ACTIONS,
    RESTOCKING_ACTIONS,
    SEMANTIC_ACTIONS,
)

from .catalog import (
    RobotTeamConfigCatalog,
    build_default_two_robot_presets,
    normalize_action_names,
)

_ALL_ACTIONS = normalize_action_names(SEMANTIC_ACTIONS)
_CATALOG_ACTIONS = normalize_action_names(
    [*SEMANTIC_ACTIONS, *FACILITY_INSPECTION_ACTIONS, *RESTOCKING_ACTIONS, *RESTAURANT_SERVING_ACTIONS]
)
ROBOT_TEAM_PRESETS: dict[str, dict[str, Any]] = build_default_two_robot_presets(_ALL_ACTIONS)
_HOTEL_FULL_ACTIONS = list(_ALL_ACTIONS)
ROBOT_TEAM_PRESETS["hotel_scaling_full_1"] = {
    "description": "Hotel robot scaling: 1 robot with the full hotel action set.",
    "robot_ids": [0],
    "robot_action_sets": {
        0: list(_HOTEL_FULL_ACTIONS),
    },
}
ROBOT_TEAM_PRESETS["hotel_scaling_full_2"] = {
    "description": "Hotel robot scaling: 2 robots, both with the full hotel action set.",
    "robot_ids": [0, 1],
    "robot_action_sets": {
        0: list(_HOTEL_FULL_ACTIONS),
        1: list(_HOTEL_FULL_ACTIONS),
    },
}
ROBOT_TEAM_PRESETS["hotel_scaling_full_3"] = {
    "description": "Hotel robot scaling: 3 homogeneous robots, each with the full hotel action set.",
    "robot_ids": [0, 1, 2],
    "robot_action_sets": {
        0: list(_HOTEL_FULL_ACTIONS),
        1: list(_HOTEL_FULL_ACTIONS),
        2: list(_HOTEL_FULL_ACTIONS),
    },
}
ROBOT_TEAM_PRESETS["hotel_scaling_full_4"] = {
    "description": "Hotel robot scaling: 4 homogeneous robots, each with the full hotel action set.",
    "robot_ids": [0, 1, 2, 3],
    "robot_action_sets": {
        0: list(_HOTEL_FULL_ACTIONS),
        1: list(_HOTEL_FULL_ACTIONS),
        2: list(_HOTEL_FULL_ACTIONS),
        3: list(_HOTEL_FULL_ACTIONS),
    },
}
ROBOT_TEAM_PRESETS["hotel_scaling_full_5"] = {
    "description": "Hotel robot scaling: 5 homogeneous robots, each with the full hotel action set.",
    "robot_ids": [0, 1, 2, 3, 4],
    "robot_action_sets": {
        0: list(_HOTEL_FULL_ACTIONS),
        1: list(_HOTEL_FULL_ACTIONS),
        2: list(_HOTEL_FULL_ACTIONS),
        3: list(_HOTEL_FULL_ACTIONS),
        4: list(_HOTEL_FULL_ACTIONS),
    },
}
ROBOT_TEAM_PRESETS["hotel_scaling_heterogeneous_3"] = ROBOT_TEAM_PRESETS["hotel_scaling_full_3"]
ROBOT_TEAM_PRESETS["hotel_scaling_heterogeneous_4"] = ROBOT_TEAM_PRESETS["hotel_scaling_full_4"]
ROBOT_TEAM_PRESETS["hotel_scaling_heterogeneous_5"] = ROBOT_TEAM_PRESETS["hotel_scaling_full_5"]
ROBOT_TEAM_PRESETS["facility_inspection_heterogeneous_3"] = {
    "description": (
        "3 robots for facility inspection: robot_0 navigation-only inspector, "
        "robot_1 mobile power-off/close specialist, robot_2 mobile power-off and carry specialist"
    ),
    "robot_ids": [0, 1, 2],
    "robot_action_sets": {
        0: ["enter_room", "navigate_to"],
        1: ["enter_room", "navigate_to", "toggle_off", "close"],
        2: ["enter_room", "navigate_to", "toggle_off", "pickup", "put_in"],
    },
}
ROBOT_TEAM_PRESETS["restocking_heterogeneous_3"] = {
    "description": (
        "3 robots for inventory restocking: robot_0 has heavy pickup capacity for apple, egg, and bread; "
        "robot_1 has medium pickup capacity for egg and bread; robot_2 has light pickup capacity for egg only."
    ),
    "robot_ids": [0, 1, 2],
    "robot_action_sets": {
        0: [
            "enter_room",
            "navigate_to",
            "open",
            "close",
            "pickup_heavy",
            "put_in",
        ],
        1: [
            "enter_room",
            "navigate_to",
            "open",
            "close",
            "pickup_medium",
            "put_in",
        ],
        2: [
            "enter_room",
            "navigate_to",
            "open",
            "close",
            "pickup_light",
            "put_in",
        ],
    },
}
ROBOT_TEAM_PRESETS["restaurant_serving_heterogeneous_3"] = {
    "description": (
        "3 robots for restaurant serving: robot_0 and robot_1 can navigate, pick up, cook, and serve; "
        "robot_2 can navigate, pick up, and serve but cannot cook."
    ),
    "robot_ids": [0, 1, 2],
    "robot_action_sets": {
        0: ["navigate_to", "pickup", "cook", "put_in"],
        1: ["navigate_to", "pickup", "cook", "put_in"],
        2: ["navigate_to", "pickup", "put_in"],
    },
}
_CATALOG = RobotTeamConfigCatalog(action_specs_or_names=_CATALOG_ACTIONS, presets=ROBOT_TEAM_PRESETS)


def list_robot_team_presets() -> list[str]:
    return _CATALOG.list_presets()


def get_robot_team_config(preset_name: str) -> dict[str, Any]:
    return _CATALOG.get_config(preset_name)
