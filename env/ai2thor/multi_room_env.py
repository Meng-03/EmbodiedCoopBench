from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from env.ai2thor.symbolic_semantic_action import ThorSymbolicEnv


@dataclass
class RoomControllerRuntime:
    runtime_key: str
    scene_name: str
    request_id: str | None = None
    init_setup_actions: list[dict[str, Any]] = field(default_factory=list)
    logical_robot_ids: list[int] = field(default_factory=list)
    logical_to_local: dict[int, int] = field(default_factory=dict)
    local_to_logical: dict[int, int] = field(default_factory=dict)
    controller_action_history: list[dict[str, Any]] = field(default_factory=list)
    semantic_action_history: list[dict[str, Any]] = field(default_factory=list)
    env: Any | None = None


class MultiRoomThorEnv:
    """Route logical robots across per-request or per-scene Thor controllers.

    By default, a scene owns one controller runtime. When multiple requests share
    the same scene, each request gets an isolated controller runtime while still
    reporting the same scene_name outward.
    """

    supports_low_level_scheduler = True

    def __init__(
        self,
        *,
        env_factory: Callable[[], Any] | None = None,
        timing_mode: str = "abstract",
        action_step_costs: dict[str, int] | None = None,
        action_step_estimators: dict[str, Callable] | None = None,
        teleport_override_path: str | None = None,
    ):
        self.timing_mode = timing_mode
        self.action_step_costs = {
            "observe": 4,
            "enter_room": 10,
            "navigate_to": 2,
            "pickup": 4,
            "put_in": 4,
            "open": 4,
            "close": 4,
            "toggle_off": 4,
            "toggle_on": 4,
            "clean": 10,
            "cook": 1,
        }
        if action_step_costs:
            self.action_step_costs.update(action_step_costs)

        self.action_step_estimators: dict[str, Callable] = {}
        if action_step_estimators:
            self.action_step_estimators.update(action_step_estimators)

        self.teleport_override_path = teleport_override_path
        self._env_factory = env_factory or self._build_default_env_factory()
        self._room_runtimes: dict[str, RoomControllerRuntime] = {}
        self._robot_runtime_keys: dict[int, str] = {}
        self._robot_request_ids: dict[int, str] = {}
        self._request_runtime_keys: dict[str, str] = {}
        self._runtime_contexts: dict[str, dict[str, Any]] = {}
        self._scene_init_setup_actions: dict[str, list[dict[str, Any]]] = {}
        self._active_runtime_key = ""
        self._active_scene_name = ""

    def _build_default_env_factory(self) -> Callable[[], ThorSymbolicEnv]:
        def _factory() -> ThorSymbolicEnv:
            return ThorSymbolicEnv(
                timing_mode=self.timing_mode,
                action_step_costs=dict(self.action_step_costs),
                action_step_estimators=dict(self.action_step_estimators),
                teleport_override_path=self.teleport_override_path,
            )

        return _factory

    @property
    def scene_name(self) -> str:
        runtime = self._get_runtime(self._active_runtime_key)
        if runtime is not None:
            return runtime.scene_name
        return self._active_scene_name

    @property
    def controller(self):
        runtime = self._get_runtime(self._active_runtime_key)
        if runtime and runtime.env is not None:
            return getattr(runtime.env, "controller", None)
        for candidate in self._room_runtimes.values():
            if candidate.env is not None:
                return getattr(candidate.env, "controller", None)
        return None

    @property
    def action_history(self) -> list[dict[str, Any]]:
        runtime = self._get_runtime(self._active_runtime_key)
        if runtime is None:
            return []
        return runtime.semantic_action_history

    def set_scene_setup(self, scene_name: str, init_setup_actions: list[dict[str, Any]] | None) -> None:
        canonical = self._room_id_to_scene_name(scene_name)
        self._scene_init_setup_actions[canonical] = list(init_setup_actions or [])
        self._runtime_contexts[canonical] = {
            "scene_name": canonical,
            "request_id": None,
            "init_setup_actions": list(init_setup_actions or []),
        }
        runtime = self._room_runtimes.get(canonical)
        if runtime is not None:
            runtime.scene_name = canonical
            runtime.request_id = None
            runtime.init_setup_actions = list(init_setup_actions or [])

    def set_scene_setups(self, scene_setups: dict[str, list[dict[str, Any]]]) -> None:
        for scene_name, init_setup_actions in scene_setups.items():
            self.set_scene_setup(scene_name, init_setup_actions)

    def register_task_context(self, task_instance: dict[str, Any] | None) -> None:
        if not isinstance(task_instance, dict):
            return
        scene_name = task_instance.get("scene_name") or task_instance.get("room_id")
        if not scene_name:
            return
        self._register_task_context(task_instance, duplicate_scene=False)

    def register_task_contexts(self, task_instances: list[dict[str, Any]]) -> None:
        """批量注册任务上下文，并决定同场景请求是否需要隔离控制器。

        规则：
        1) 先统计每个 canonical scene 出现次数。
        2) 若同一 scene 出现多次，则该 scene 下的每个 request 使用独立 runtime_key。
        3) 若 scene 仅出现一次，则继续复用 scene 级 runtime_key。
        """
        # 只保留有效的 task 字典，避免上游混入空值或非 dict 数据。
        normalized_tasks = [task for task in task_instances if isinstance(task, dict)]

        # 第一遍：统计规范化场景名的出现次数，用于判断是否“同场景多请求”。
        scene_counts: dict[str, int] = {}
        for task_instance in normalized_tasks:
            scene_name = task_instance.get("scene_name") or task_instance.get("room_id")
            if not scene_name:
                continue
            canonical = self._room_id_to_scene_name(str(scene_name))
            scene_counts[canonical] = scene_counts.get(canonical, 0) + 1

        # 第二遍：逐任务注册上下文，并按 scene 重复情况决定是否 request 级隔离。
        for task_instance in normalized_tasks:
            scene_name = task_instance.get("scene_name") or task_instance.get("room_id")
            if not scene_name:
                continue
            canonical = self._room_id_to_scene_name(str(scene_name))
            shared_scene_runtime = bool(task_instance.get("shared_scene_runtime"))
            self._register_task_context(
                task_instance,
                duplicate_scene=scene_counts.get(canonical, 0) > 1 and not shared_scene_runtime,
            )

    def set_robot_request_assignments(self, assignments: dict[int, str]) -> None:
        self._robot_request_ids = {
            int(robot_id): str(request_id)
            for robot_id, request_id in (assignments or {}).items()
            if request_id
        }

    def enter_room(
        self,
        logical_robot_id: int,
        room_id: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        scene_name = self._room_id_to_scene_name(room_id)
        runtime_key = self._resolve_runtime_key(
            scene_name=scene_name,
            robot_id=logical_robot_id,
            request_id=request_id,
        )
        previous_runtime_key = self._robot_runtime_keys.get(logical_robot_id)

        if request_id:
            self._robot_request_ids[logical_robot_id] = str(request_id)

        if previous_runtime_key == runtime_key:
            runtime = self._require_runtime(runtime_key, scene_name=scene_name)
            if logical_robot_id not in runtime.logical_robot_ids:
                runtime.logical_robot_ids.append(logical_robot_id)
                runtime.logical_robot_ids.sort()
                self._rebuild_runtime(runtime_key)
            self._active_runtime_key = runtime_key
            self._active_scene_name = scene_name
            return self.get_agent_observation(logical_robot_id)

        if previous_runtime_key:
            previous_runtime = self._require_runtime(previous_runtime_key)
            previous_runtime.logical_robot_ids = [
                rid for rid in previous_runtime.logical_robot_ids if rid != logical_robot_id
            ]
            previous_runtime.logical_to_local.pop(logical_robot_id, None)
            if previous_runtime.logical_robot_ids:
                self._rebuild_runtime(previous_runtime_key)
            else:
                previous_runtime.logical_to_local = {}
                previous_runtime.local_to_logical = {}

        target_runtime = self._require_runtime(runtime_key, scene_name=scene_name)
        if logical_robot_id not in target_runtime.logical_robot_ids:
            target_runtime.logical_robot_ids.append(logical_robot_id)
            target_runtime.logical_robot_ids.sort()

        self._robot_runtime_keys[logical_robot_id] = runtime_key
        self._rebuild_runtime(runtime_key)
        self._active_runtime_key = runtime_key
        self._active_scene_name = scene_name
        return self.get_agent_observation(logical_robot_id)

    def get_scene_name(self, agent_id: int | None = None, room_id: str | None = None) -> str:
        if room_id:
            return self._room_id_to_scene_name(room_id)
        if agent_id is not None:
            runtime = self._get_runtime(self._robot_runtime_keys.get(agent_id))
            if runtime is not None:
                return runtime.scene_name
            return ""
        return self._active_scene_name

    def get_room_observation(
        self,
        room_id: str,
        *,
        request_id: str | None = None,
        robot_id: int | None = None,
    ) -> dict[str, Any]:
        scene_name = self._room_id_to_scene_name(room_id)
        runtime = self._resolve_runtime_for_observation(
            scene_name=scene_name,
            request_id=request_id,
            robot_id=robot_id,
        )
        if runtime is None:
            return self._empty_observation(scene_name)
        if runtime.logical_robot_ids:
            return self.get_agent_observation(runtime.logical_robot_ids[0])
        runtime = self._ensure_runtime_observation_env(
            runtime.runtime_key,
            scene_name=runtime.scene_name,
            force_rebuild=True,
        )
        if runtime.env is not None and hasattr(runtime.env, "get_observation"):
            return self._decorate_observation(runtime, runtime.env.get_observation(agent_id=0))
        return self._empty_observation(scene_name, history=runtime.semantic_action_history)

    def get_request_observation(self, request_id: str) -> dict[str, Any]:
        runtime_key = self._request_runtime_keys.get(str(request_id))
        if not runtime_key:
            return self._empty_observation("")
        runtime = self._get_runtime(runtime_key)
        if runtime is None:
            context = self._runtime_contexts.get(runtime_key) or {}
            if not context:
                return self._empty_observation("")
            runtime = self._ensure_runtime_observation_env(
                runtime_key,
                scene_name=str(context.get("scene_name") or ""),
            )
        elif runtime.env is None:
            runtime = self._ensure_runtime_observation_env(
                runtime_key,
                scene_name=runtime.scene_name,
            )
        if runtime.logical_robot_ids:
            return self.get_agent_observation(runtime.logical_robot_ids[0])
        runtime = self._ensure_runtime_observation_env(
            runtime_key,
            scene_name=runtime.scene_name,
            force_rebuild=True,
        )
        if runtime.env is not None and hasattr(runtime.env, "get_observation"):
            return self._decorate_observation(runtime, runtime.env.get_observation(agent_id=0))
        if runtime.env is not None and hasattr(runtime.env, "get_agent_observation"):
            return self._decorate_observation(runtime, runtime.env.get_agent_observation(agent_id=0))
        return self._empty_observation(runtime.scene_name, history=runtime.semantic_action_history)

    def get_room_action_history(self, room_id: str) -> list[dict[str, Any]]:
        runtime = self._resolve_runtime_for_observation(self._room_id_to_scene_name(room_id))
        if runtime is None:
            return []
        return list(runtime.semantic_action_history)

    def get_agent_observation(self, agent_id: int = 0):
        resolved = self._resolve_runtime_for_robot(agent_id)
        if resolved is None:
            return self._empty_observation(
                "",
                warning=f"robot_{agent_id} has not entered any room yet",
            )
        runtime, local_agent_id = resolved
        obs = runtime.env.get_agent_observation(local_agent_id)
        self._active_runtime_key = runtime.runtime_key
        self._active_scene_name = runtime.scene_name
        return self._decorate_observation(runtime, obs)

    def get_observation(self, agent_id: int = 0):
        return self.get_agent_observation(agent_id=agent_id)

    def append_semantic_action_history(
        self,
        *,
        agent_id: int,
        semantic_action: dict[str, Any],
        success: bool,
        error_msg: str = "",
        planned_steps: int | None = None,
        done_steps: int | None = None,
    ) -> None:
        resolved = self._resolve_runtime_for_robot(agent_id)
        if resolved is None:
            return
        runtime, _ = resolved
        logged_action = dict(semantic_action)
        logged_action["step_success"] = success
        if planned_steps is not None:
            logged_action["planned_steps"] = int(planned_steps)
        if done_steps is not None:
            logged_action["done_steps"] = int(done_steps)
        if error_msg:
            logged_action["step_error"] = error_msg
        runtime.semantic_action_history.append(logged_action)
        if runtime.env is not None and hasattr(runtime.env, "action_history"):
            runtime.env.action_history = list(runtime.semantic_action_history)

    def _execute_thor_action(self, action_name, agent_id: int = 0, **kwargs):
        resolved = self._require_runtime_for_robot(agent_id)
        runtime, local_agent_id = resolved
        success, error_msg = runtime.env._execute_thor_action(action_name, agent_id=local_agent_id, **kwargs)
        if success:
            runtime.controller_action_history.append(
                {
                    "method": "_execute_thor_action",
                    "logical_robot_id": agent_id,
                    "action_name": action_name,
                    "kwargs": dict(kwargs),
                }
            )
        self._active_runtime_key = runtime.runtime_key
        self._active_scene_name = runtime.scene_name
        return success, error_msg

    def _record_controller_teleport_full(
        self,
        *,
        logical_robot_id: int,
        position: dict[str, Any],
        rotation: dict[str, Any],
        horizon: float,
        standing: bool = True,
    ) -> None:
        resolved = self._require_runtime_for_robot(logical_robot_id)
        runtime, _ = resolved
        runtime.controller_action_history.append(
            {
                "method": "_teleport_full",
                "logical_robot_id": logical_robot_id,
                "position": dict(position),
                "rotation": dict(rotation),
                "horizon": horizon,
                "standing": standing,
            }
        )

    def _place_object_at_point(self, holding_obj_id, agent_id: int = 0):
        runtime, local_agent_id = self._require_runtime_for_robot(agent_id)
        success, error_msg = runtime.env._place_object_at_point(holding_obj_id, agent_id=local_agent_id)
        if success:
            runtime.controller_action_history.append(
                {
                    "method": "_place_object_at_point",
                    "logical_robot_id": agent_id,
                    "holding_object_id": holding_obj_id,
                }
            )
        self._active_runtime_key = runtime.runtime_key
        self._active_scene_name = runtime.scene_name
        return success, error_msg

    def _teleport_near_object(self, obj_id, agent_id: int = 0):
        runtime, local_agent_id = self._require_runtime_for_robot(agent_id)
        success, error_msg, estimated_steps, step_consumed = runtime.env._teleport_near_object(
            obj_id,
            agent_id=local_agent_id,
        )
        if success:
            runtime.controller_action_history.append(
                {
                    "method": "_teleport_near_object",
                    "logical_robot_id": agent_id,
                    "object_id": obj_id,
                }
            )
        self._active_runtime_key = runtime.runtime_key
        self._active_scene_name = runtime.scene_name
        return success, error_msg, estimated_steps, step_consumed

    def _apply_done_steps(self, step_count: int, agent_id: int = 0):
        runtime, local_agent_id = self._require_runtime_for_robot(agent_id)
        self._active_runtime_key = runtime.runtime_key
        self._active_scene_name = runtime.scene_name
        return runtime.env._apply_done_steps(step_count, agent_id=local_agent_id)

    def _observe_surroundings(self, agent_id: int = 0):
        runtime, local_agent_id = self._require_runtime_for_robot(agent_id)
        success, error_msg, observation, rotate_steps = runtime.env._observe_surroundings(agent_id=local_agent_id)
        self._active_runtime_key = runtime.runtime_key
        self._active_scene_name = runtime.scene_name
        return success, error_msg, self._decorate_observation(runtime, observation), rotate_steps

    def _get_holding_object_id(self, agent_id: int = 0):
        runtime, local_agent_id = self._require_runtime_for_robot(agent_id)
        self._active_runtime_key = runtime.runtime_key
        self._active_scene_name = runtime.scene_name
        return runtime.env._get_holding_object_id(agent_id=local_agent_id)

    def _should_place_object_at_origin(self, holding_obj_id) -> bool:
        return ThorSymbolicEnv._should_place_object_at_origin(holding_obj_id)

    def _resolve_action_step_cost(
        self,
        action_name: str,
        semantic_action: dict,
        agent_id: int,
        success: bool,
        estimated_steps: int | None = None,
    ) -> int:
        resolved = self._resolve_runtime_for_robot(agent_id)
        if resolved is not None:
            runtime, local_agent_id = resolved
            if hasattr(runtime.env, "_resolve_action_step_cost"):
                return int(
                    runtime.env._resolve_action_step_cost(
                        action_name=action_name,
                        semantic_action=semantic_action,
                        agent_id=local_agent_id,
                        success=success,
                        estimated_steps=estimated_steps,
                    )
                )
        if estimated_steps is not None:
            return max(int(estimated_steps), 1)
        normalized = action_name.replace("_thor", "") if action_name.endswith("_thor") else action_name
        return max(int(self.action_step_costs.get(normalized, 1)), 1)

    def _room_id_to_scene_name(self, room_id: str) -> str:
        room_text = str(room_id).strip()
        if not room_text:
            raise ValueError("room_id cannot be empty")
        if room_text.startswith("FloorPlan"):
            return room_text
        match = re.search(r"(\d+)$", room_text)
        if not match:
            raise ValueError(
                f"Cannot parse scene id from room_id={room_id!r}; expected Room401, 401, or FloorPlan401"
            )
        return f"FloorPlan{match.group(1)}"

    def _get_runtime(self, runtime_key: str | None) -> RoomControllerRuntime | None:
        if not runtime_key:
            return None
        return self._room_runtimes.get(runtime_key)

    def _require_runtime(
        self,
        runtime_key: str,
        *,
        scene_name: str | None = None,
    ) -> RoomControllerRuntime:
        runtime = self._room_runtimes.get(runtime_key)
        if runtime is not None:
            return runtime
        context = self._runtime_contexts.get(runtime_key) or {}
        resolved_scene_name = str(context.get("scene_name") or scene_name or runtime_key)
        runtime = RoomControllerRuntime(
            runtime_key=runtime_key,
            scene_name=resolved_scene_name,
            request_id=context.get("request_id"),
            init_setup_actions=list(
                context.get("init_setup_actions")
                or self._scene_init_setup_actions.get(resolved_scene_name, [])
            ),
        )
        self._room_runtimes[runtime_key] = runtime
        return runtime

    def _resolve_runtime_for_robot(self, logical_robot_id: int) -> tuple[RoomControllerRuntime, int] | None:
        runtime_key = self._robot_runtime_keys.get(logical_robot_id)
        if not runtime_key:
            return None
        runtime = self._room_runtimes.get(runtime_key)
        if runtime is None or runtime.env is None:
            return None
        local_agent_id = runtime.logical_to_local.get(logical_robot_id)
        if local_agent_id is None:
            return None
        return runtime, local_agent_id

    def _require_runtime_for_robot(self, logical_robot_id: int) -> tuple[RoomControllerRuntime, int]:
        resolved = self._resolve_runtime_for_robot(logical_robot_id)
        if resolved is None:
            raise RuntimeError(
                f"robot_{logical_robot_id} has not entered a room yet; call enter_room first"
            )
        return resolved

    @staticmethod
    def _stop_env_controller(room_env: Any | None) -> None:
        if room_env is None:
            return
        controller = getattr(room_env, "controller", None)
        if controller is None or not hasattr(controller, "stop"):
            return
        try:
            controller.stop()
        except Exception:
            pass

    def _rebuild_runtime(self, runtime_key: str) -> None:
        runtime = self._require_runtime(runtime_key)
        desired_robot_ids = sorted(runtime.logical_robot_ids)
        if not desired_robot_ids:
            runtime.logical_to_local = {}
            runtime.local_to_logical = {}
            return

        previous_env = runtime.env
        previous_logical_to_local = dict(runtime.logical_to_local)
        previous_local_to_logical = dict(runtime.local_to_logical)

        replay_robot_ids = sorted(
            {
                *desired_robot_ids,
                *[
                    int(record.get("logical_robot_id"))
                    for record in (runtime.controller_action_history or [])
                    if record.get("logical_robot_id") is not None
                ],
            }
        )

        room_env = self._env_factory()
        if hasattr(room_env, "init_setup_actions"):
            room_env.init_setup_actions = list(runtime.init_setup_actions)
        room_env.reset(scene_name=runtime.scene_name, agentcount=len(replay_robot_ids))

        replay_logical_to_local = {
            logical_robot_id: local_agent_id
            for local_agent_id, logical_robot_id in enumerate(replay_robot_ids)
        }
        replay_local_to_logical = {
            local_agent_id: logical_robot_id
            for logical_robot_id, local_agent_id in replay_logical_to_local.items()
        }

        runtime.env = room_env
        runtime.logical_to_local = replay_logical_to_local
        runtime.local_to_logical = replay_local_to_logical

        try:
            for record in runtime.controller_action_history:
                self._replay_controller_action(runtime, record)

            runtime.logical_to_local = {
                logical_robot_id: replay_logical_to_local[logical_robot_id]
                for logical_robot_id in desired_robot_ids
            }
            runtime.local_to_logical = {
                local_agent_id: logical_robot_id
                for logical_robot_id, local_agent_id in runtime.logical_to_local.items()
            }

            if hasattr(room_env, "action_history"):
                room_env.action_history = list(runtime.semantic_action_history)
        except Exception:
            self._stop_env_controller(room_env)
            runtime.env = previous_env
            runtime.logical_to_local = previous_logical_to_local
            runtime.local_to_logical = previous_local_to_logical
            raise

        if previous_env is not None and previous_env is not room_env:
            self._stop_env_controller(previous_env)

    def _ensure_runtime_observation_env(
        self,
        runtime_key: str,
        *,
        scene_name: str | None = None,
        force_rebuild: bool = False,
    ) -> RoomControllerRuntime:
        runtime = self._require_runtime(runtime_key, scene_name=scene_name)
        if runtime.env is not None and not force_rebuild:
            return runtime

        previous_env = runtime.env
        previous_logical_to_local = dict(runtime.logical_to_local)
        previous_local_to_logical = dict(runtime.local_to_logical)

        replay_robot_ids = sorted(
            {
                int(record.get("logical_robot_id"))
                for record in (runtime.controller_action_history or [])
                if record.get("logical_robot_id") is not None
            }
        )
        if not replay_robot_ids:
            replay_robot_ids = [0]

        room_env = self._env_factory()
        if hasattr(room_env, "init_setup_actions"):
            room_env.init_setup_actions = list(runtime.init_setup_actions)

        try:
            room_env.reset(scene_name=runtime.scene_name, agentcount=len(replay_robot_ids))
            runtime.env = room_env
            runtime.logical_to_local = {
                logical_robot_id: local_agent_id
                for local_agent_id, logical_robot_id in enumerate(replay_robot_ids)
            }
            runtime.local_to_logical = {
                local_agent_id: logical_robot_id
                for logical_robot_id, local_agent_id in runtime.logical_to_local.items()
            }
            for record in runtime.controller_action_history:
                self._replay_controller_action(runtime, record)
            if hasattr(room_env, "action_history"):
                room_env.action_history = list(runtime.semantic_action_history)
        except Exception:
            self._stop_env_controller(room_env)
            runtime.env = previous_env
            runtime.logical_to_local = previous_logical_to_local
            runtime.local_to_logical = previous_local_to_logical
            raise

        runtime.env = room_env
        runtime.logical_to_local = {}
        runtime.local_to_logical = {}
        if previous_env is not None and previous_env is not room_env:
            self._stop_env_controller(previous_env)
        return runtime

    def _replay_controller_action(self, runtime: RoomControllerRuntime, record: dict[str, Any]) -> None:
        logical_robot_id = int(record.get("logical_robot_id", 0))
        local_agent_id = runtime.logical_to_local.get(logical_robot_id)
        if local_agent_id is None:
            return

        method = record.get("method")
        if method == "_execute_thor_action":
            runtime.env._execute_thor_action(
                record.get("action_name"),
                agent_id=local_agent_id,
                **dict(record.get("kwargs") or {}),
            )
            return

        if method == "_teleport_near_object":
            runtime.env._teleport_near_object(
                record.get("object_id"),
                agent_id=local_agent_id,
            )
            return

        if method == "_teleport_full":
            runtime.env.controller.step(
                action="TeleportFull",
                agentId=local_agent_id,
                position=dict(record.get("position") or {}),
                rotation=dict(record.get("rotation") or {}),
                horizon=float(record.get("horizon", 0.0)),
                standing=bool(record.get("standing", True)),
            )
            return

        if method == "_place_object_at_point":
            runtime.env._place_object_at_point(
                record.get("holding_object_id"),
                agent_id=local_agent_id,
            )

    def _decorate_observation(
        self,
        runtime: RoomControllerRuntime,
        observation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(observation, dict):
            return self._empty_observation(runtime.scene_name, history=runtime.semantic_action_history)
        decorated = dict(observation)
        decorated["scene_name"] = runtime.scene_name
        decorated["request_id"] = runtime.request_id
        decorated["runtime_key"] = runtime.runtime_key
        decorated["history"] = list(runtime.semantic_action_history)
        return decorated

    def _register_task_context(
        self,
        task_instance: dict[str, Any],
        *,
        duplicate_scene: bool,
    ) -> None:
        scene_name = task_instance.get("scene_name") or task_instance.get("room_id")
        if not scene_name:
            return
        canonical = self._room_id_to_scene_name(str(scene_name))
        request_id = task_instance.get("request_id")
        runtime_key = str(request_id) if duplicate_scene and request_id else canonical
        init_setup_actions = list(task_instance.get("init_setup_actions") or [])

        self._runtime_contexts[runtime_key] = {
            "scene_name": canonical,
            "request_id": str(request_id) if request_id else None,
            "init_setup_actions": init_setup_actions,
        }
        if request_id:
            self._request_runtime_keys[str(request_id)] = runtime_key
        if runtime_key == canonical:
            self._scene_init_setup_actions[canonical] = init_setup_actions

        runtime = self._room_runtimes.get(runtime_key)
        if runtime is not None:
            runtime.scene_name = canonical
            runtime.request_id = str(request_id) if request_id else None
            runtime.init_setup_actions = init_setup_actions

    def _resolve_runtime_key(
        self,
        *,
        scene_name: str,
        robot_id: int | None = None,
        request_id: str | None = None,
    ) -> str:
        resolved_request_id = request_id
        if not resolved_request_id and robot_id is not None:
            resolved_request_id = self._robot_request_ids.get(robot_id)
        if resolved_request_id:
            return self._request_runtime_keys.get(str(resolved_request_id), scene_name)
        return scene_name

    def _resolve_runtime_for_observation(
        self,
        scene_name: str,
        *,
        request_id: str | None = None,
        robot_id: int | None = None,
    ) -> RoomControllerRuntime | None:
        runtime_key = self._resolve_runtime_key(
            scene_name=scene_name,
            robot_id=robot_id,
            request_id=request_id,
        )
        runtime = self._get_runtime(runtime_key)
        if runtime is not None:
            return runtime

        matches = [candidate for candidate in self._room_runtimes.values() if candidate.scene_name == scene_name]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _empty_observation(
        scene_name: str,
        *,
        history: list[dict[str, Any]] | None = None,
        warning: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {"objects": []}
        if warning:
            metadata["agentObservationWarning"] = warning
        return {
            "image": None,
            "metadata": metadata,
            "history": list(history or []),
            "scene_name": scene_name,
        }
