from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENE_ORIGIN_DIR = _REPO_ROOT / "env" / "ai2thor" / "scene_origin"
DEFAULT_REQUEST_BUNDLE_FILE_PREFIX = "bundles"


def _is_usedup_toiletpaper_asset(asset_id: str | None) -> bool:
    return "used_up" in str(asset_id or "").strip().lower()


@lru_cache(maxsize=64)
def _load_scene_toiletpaper_asset_map(scene_name: str) -> dict[str, str]:
    scene_file = _SCENE_ORIGIN_DIR / f"{scene_name}.json"
    if not scene_file.exists():
        return {}

    with scene_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list):
        return {}

    return {
        str(obj.get("objectId")): str(obj.get("assetId") or "")
        for obj in objects
        if isinstance(obj, dict) and str(obj.get("objectType") or "").lower() == "toiletpaper"
    }


def _swap_exact_string_values(payload: Any, first: str, second: str) -> Any:
    if isinstance(payload, dict):
        return {key: _swap_exact_string_values(value, first, second) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_swap_exact_string_values(item, first, second) for item in payload]
    if isinstance(payload, str):
        if payload == first:
            return second
        if payload == second:
            return first
    return payload


@dataclass
class RequestSamplingSpec:
    request_count: int = 1
    rooms: list[str] | None = None
    total_active_objects: list[int] | None = None
    pddl_plan_lengths: list[int] | None = None
    sample_seed: int | None = None
    sample_mode: str = "random"  # random | sequential


class TaskAndTeamSetup:
    """Eval-oriented setup helper for request selection and robot-team selection.

    Responsibilities:
    1) Load existing sampled bundle JSON and return selected bundle.
    2) Resolve robot-team preset.
    3) Provide task lookup/index utilities for eval flow.

    Note:
    Online request sampling in this helper is deprecated/disabled.
    Eval reads the fixed benchmark bundles from tasks/bundles.
    """

    def __init__(
        self,
        task_json_path: str | Path,
        get_robot_team_config_fn: Callable[[str], dict[str, Any]],
        list_robot_team_presets_fn: Callable[[], list[str]],
    ) -> None:
        self.task_json_path = Path(task_json_path)
        self._get_robot_team_config = get_robot_team_config_fn
        self._list_robot_team_presets = list_robot_team_presets_fn
        if not self.task_json_path.exists():
            raise FileNotFoundError(f"task_json not found: {self.task_json_path}")

        # Lazy caches: bundle-only eval path does not need task indexing at init time.
        self.tasks_by_room: dict[str, list[dict[str, Any]]] = {}
        self._all_entries: list[dict[str, Any]] = []
        self._task_by_instance_id: dict[str, dict[str, Any]] = {}
        self._task_index_ready = False

    def _ensure_task_index(self) -> None:
        if self._task_index_ready:
            return

        with self.task_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid task JSON: expected dict room->tasks, got {type(data)}")

        tasks_by_room: dict[str, list[dict[str, Any]]] = {}
        all_entries: list[dict[str, Any]] = []
        task_by_instance_id: dict[str, dict[str, Any]] = {}

        for room, tasks in sorted(data.items(), key=lambda x: x[0]):
            if not isinstance(tasks, list):
                continue
            room_tasks: list[dict[str, Any]] = []
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                task = self.normalize_task_instance(task)
                instance_id = str(task.get("instance_id", "")).strip()
                if not instance_id:
                    continue

                total_active = int(task.get("total_active_objects", 0) or 0)
                pddl_len = int(task.get("pddl_plan_length", 0) or 0)
                total_active_type = task.get("total_active_objects_type")
                if not isinstance(total_active_type, dict):
                    total_active_type = {}

                entry = {
                    "instance_id": instance_id,
                    "request_id": f"req_{instance_id}",
                    "scene_name": str(task.get("scene_name") or room),
                    "room_id": str(task.get("scene_name") or room).replace("FloorPlan", ""),
                    "total_active_objects": total_active,
                    "total_active_objects_type": total_active_type,
                    "pddl_plan_length": pddl_len,
                    "object_mapping": task.get("id_mapping", {}),
                    "task": task,
                }
                room_tasks.append(task)
                all_entries.append(entry)
                task_by_instance_id[instance_id] = task

            tasks_by_room[room] = room_tasks

        if not all_entries:
            raise ValueError("No valid tasks found in task JSON.")

        self.tasks_by_room = tasks_by_room
        self._all_entries = all_entries
        self._task_by_instance_id = task_by_instance_id
        self._task_index_ready = True

    @staticmethod
    def normalize_task_instance(task: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(task, dict):
            return task

        scene_name = str(task.get("scene_name") or "")
        id_mapping = task.get("id_mapping") or {}
        spent_id = id_mapping.get("toiletpaper_spent")
        new_id = id_mapping.get("toiletpaper_new")
        if not scene_name or not spent_id or not new_id:
            return task

        asset_map = _load_scene_toiletpaper_asset_map(scene_name)
        spent_asset = asset_map.get(str(spent_id), "")
        new_asset = asset_map.get(str(new_id), "")
        if _is_usedup_toiletpaper_asset(spent_asset) or not _is_usedup_toiletpaper_asset(new_asset):
            return task

        return _swap_exact_string_values(deepcopy(task), str(spent_id), str(new_id))

    def list_robot_team_presets(self) -> list[str]:
        return self._list_robot_team_presets()

    @staticmethod
    def _sampling_disabled_error() -> RuntimeError:
        return RuntimeError(
            "Online request sampling is disabled. "
            "Please read pre-sampled req bundle files from the configured request_bundle_dir."
        )

    def get_robot_team(self, preset_name: str) -> dict[str, Any]:
        return self._get_robot_team_config(preset_name)

    def get_sampling_space(self) -> dict[str, Any]:
        self._ensure_task_index()
        rooms = sorted({e["scene_name"] for e in self._all_entries})
        total_active_values = sorted({int(e["total_active_objects"]) for e in self._all_entries})

        nested_counts: dict[str, dict[str, int]] = {}
        for active in total_active_values:
            per_plan: dict[str, int] = {}
            for e in self._all_entries:
                if int(e["total_active_objects"]) != active:
                    continue
                pddl_len = str(int(e["pddl_plan_length"]))
                per_plan[pddl_len] = per_plan.get(pddl_len, 0) + 1
            nested_counts[str(active)] = dict(sorted(per_plan.items(), key=lambda x: int(x[0])))

        return {
            "task_json_path": str(self.task_json_path),
            "total_requests": len(self._all_entries),
            "request_count_space": {
                "min": 1,
                "max": len(self._all_entries),
            },
            "room_space": rooms,
            "total_active_objects_space": total_active_values,
            "total_active_objects_to_pddl_plan_count": nested_counts,
        }

    def sample_request_entries(self, spec: RequestSamplingSpec) -> list[dict[str, Any]]:
        raise self._sampling_disabled_error()

    def build_request_bundle(
        self,
        spec: RequestSamplingSpec,
        bundle_id: str = "bundle_001",
    ) -> dict[str, Any]:
        raise self._sampling_disabled_error()

    def get_detailed_sampling_space(self) -> dict[str, Any]:
        self._ensure_task_index()
        rooms = sorted({e["scene_name"] for e in self._all_entries})
        total_active_values = sorted({int(e["total_active_objects"]) for e in self._all_entries})

        active_objects_space: dict[str, list[dict[str, Any]]] = {}
        for active in total_active_values:
            plan_to_ids: dict[int, list[str]] = {}
            for e in self._all_entries:
                if int(e["total_active_objects"]) != active:
                    continue
                pddl_len = int(e["pddl_plan_length"])
                plan_to_ids.setdefault(pddl_len, []).append(e["instance_id"])

            groups = [
                {
                    "pddl_plan_length": pddl_len,
                    "task_count": len(ids),
                    "instance_ids": ids,
                }
                for pddl_len, ids in sorted(plan_to_ids.items())
            ]
            active_objects_space[str(active)] = groups

        return {
            "task_json_path": str(self.task_json_path),
            "total_requests": len(self._all_entries),
            "request_count_space": {
                "min": 1,
                "max": len(self._all_entries),
            },
            "room_space": rooms,
            "active_objects_space": active_objects_space,
        }

    def export_sampling_space(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.get_sampling_space(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out

    def export_detailed_sampling_space(self, output_path: str | Path) -> Path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.get_detailed_sampling_space(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out

    def export_request_bundles(
        self,
        output_path: str | Path,
        spec: RequestSamplingSpec,
        bundle_count: int = 1,
    ) -> Path:
        raise self._sampling_disabled_error()

    def get_task_by_instance_id(self, instance_id: str) -> dict[str, Any]:
        self._ensure_task_index()
        if instance_id not in self._task_by_instance_id:
            raise KeyError(f"Unknown instance_id: {instance_id}")
        return self._task_by_instance_id[instance_id]

    def load_request_bundles_by_counts(
        self,
        bundle_dir: str | Path,
        request_counts: list[int],
        bundles_per_count: int = 1,
        bundle_file_prefix: str = DEFAULT_REQUEST_BUNDLE_FILE_PREFIX,
    ) -> list[dict[str, Any]]:
        """Load bundles from req{count} files under one directory.

        Expected file name: {bundle_file_prefix}_req{count}.json
        """
        if bundles_per_count <= 0:
            raise ValueError("bundles_per_count must be > 0")

        base_dir = Path(bundle_dir)
        loaded: list[dict[str, Any]] = []

        for count in sorted(set(int(c) for c in request_counts)):
            if count <= 0:
                raise ValueError(f"request_count must be > 0, got {count}")

            fp = base_dir / f"{bundle_file_prefix}_req{count}.json"
            data = self.load_bundles_file(fp)
            bundles = data.get("bundles")
            if not isinstance(bundles, list) or not bundles:
                raise ValueError(f"Invalid bundles file or empty bundles: {fp}")

            take_n = min(bundles_per_count, len(bundles))
            for idx in range(take_n):
                b = bundles[idx]
                if not isinstance(b, dict):
                    continue
                loaded.append(
                    {
                        "request_count": count,
                        "source_file": str(fp),
                        "bundle_index": idx,
                        "bundle": b,
                    }
                )

        return loaded

    def build_request_robot_combinations(
        self,
        request_bundle_items: list[dict[str, Any]],
        robot_team_presets: list[str],
    ) -> list[dict[str, Any]]:
        """Build cartesian combinations between request bundles and robot presets."""
        combos: list[dict[str, Any]] = []
        combo_idx = 0

        for item in request_bundle_items:
            bundle = item.get("bundle")
            if not isinstance(bundle, dict):
                continue
            for preset in robot_team_presets:
                team = self.get_robot_team(preset)
                combo_idx += 1
                combos.append(
                    {
                        "combo_id": f"combo_{combo_idx:05d}",
                        "request_count": int(item.get("request_count", 0) or 0),
                        "source_file": str(item.get("source_file", "")),
                        "bundle_index": int(item.get("bundle_index", 0) or 0),
                        "bundle_id": str(bundle.get("bundle_id", "")),
                        "request_ids": list(bundle.get("request_ids", [])),
                        "bundle_total_active_objects": int(bundle.get("bundle_total_active_objects", 0) or 0),
                        "bundle_total_pddl_plan_length": int(bundle.get("bundle_total_pddl_plan_length", 0) or 0),
                        "robot_team_preset": preset,
                        "robot_team": team,
                    }
                )

        return combos

    @staticmethod
    def load_bundles_file(bundles_path: str | Path) -> dict[str, Any]:
        path = Path(bundles_path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("bundles file must be a JSON object")
        return data

    @staticmethod
    def load_request_bundle(
        bundles_path: str | Path,
        bundle_index: int = 0,
        bundle_id: str | None = None,
    ) -> dict[str, Any]:
        """Load one request bundle from saved JSON.

        Supports:
        1) Legacy format: {"bundles": [...]}.
        2) Profiled format: {"profiles": [{"bundles": [...]}, ...]}.

        If ``bundle_id`` is provided it takes priority over ``bundle_index``.
        """
        data = TaskAndTeamSetup.load_bundles_file(bundles_path)
        bundles = data.get("bundles")

        if not isinstance(bundles, list) or not bundles:
            profiles = data.get("profiles")
            flattened: list[dict[str, Any]] = []
            if isinstance(profiles, list):
                for profile in profiles:
                    if not isinstance(profile, dict):
                        continue
                    pb = profile.get("bundles")
                    if isinstance(pb, list):
                        for item in pb:
                            if isinstance(item, dict):
                                flattened.append(item)
            bundles = flattened

        if not isinstance(bundles, list) or not bundles:
            raise ValueError(
                "Invalid bundles file: expected non-empty 'bundles' or 'profiles[*].bundles'"
            )

        if bundle_id is not None:
            for item in bundles:
                if isinstance(item, dict) and item.get("bundle_id") == bundle_id:
                    return item
            raise ValueError(f"bundle_id not found: {bundle_id!r} (file={bundles_path})")

        if bundle_index < 0 or bundle_index >= len(bundles):
            raise IndexError(f"bundle_index out of range: {bundle_index}, total={len(bundles)}")
        bundle = bundles[bundle_index]
        if not isinstance(bundle, dict):
            raise ValueError("Invalid bundles file: bundle item must be an object")
        return bundle

    def resolve_eval_selection_from_config(
        self,
        config_path: str | Path,
        profile_name: str,
        request_count: int,
        sample_seed: int | None = None,
        sample_mode: str = "random",
    ) -> dict[str, Any]:
        raise self._sampling_disabled_error()

    def resolve_eval_selection(
        self,
        robot_team_preset: str,
        request_bundle_path: str | Path | None = None,
        request_bundle_index: int = 0,
        request_spec: RequestSamplingSpec | None = None,
    ) -> dict[str, Any]:
        """Resolve eval inputs from saved bundle only."""
        robot_team = self.get_robot_team(robot_team_preset)

        if request_bundle_path is not None:
            bundle = self.load_request_bundle(request_bundle_path, bundle_index=request_bundle_index)
        else:
            raise self._sampling_disabled_error()

        return {
            "request_bundle": bundle,
            "robot_team_preset": robot_team_preset,
            "robot_team": robot_team,
        }
