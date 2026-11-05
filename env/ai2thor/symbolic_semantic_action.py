# env/ai2thor/symbolic_semantic_action.py
from contextlib import contextmanager
import json
import math
import re
import signal
import threading
import time
from pathlib import Path
from typing import Callable

from ai2thor.controller import Controller
import numpy as np
from PIL import Image, ImageDraw
import env.ai2thor.constants as constants
from env.ai2thor.utils.thor_utils import *

class ThorSymbolicEnv:
    def __init__(
        self,
        init_setup_actions=None,
        timing_mode: str = "abstract",
        action_step_costs: dict[str, int] | None = None,
        action_step_estimators: dict[str, Callable] | None = None,
        teleport_override_path: str | None = None,
        controller_reset_retries: int = constants.THOR_CONTROLLER_RESET_RETRIES,
        controller_restart_backoff_sec: float = constants.THOR_CONTROLLER_RESTART_BACKOFF_SEC,
        controller_reset_timeout_sec: float = constants.THOR_CONTROLLER_RESET_TIMEOUT_SEC,
        controller_reset_timeout_retries: int = constants.THOR_CONTROLLER_RESET_TIMEOUT_RETRIES,
        controller_timeout_restart_backoff_sec: float = constants.THOR_CONTROLLER_TIMEOUT_RESTART_BACKOFF_SEC,
    ):
        # 启动 AI2-THOR (初始不加载任何特定场景，或者随意加载一个)
        self.scene_name = ""
        self.controller_reset_retries = max(1, int(controller_reset_retries))
        self.controller_restart_backoff_sec = max(0.0, float(controller_restart_backoff_sec))
        self.controller_reset_timeout_sec = max(0.0, float(controller_reset_timeout_sec))
        self.controller_reset_timeout_retries = max(0, int(controller_reset_timeout_retries))
        self.controller_timeout_restart_backoff_sec = max(
            0.0,
            float(controller_timeout_restart_backoff_sec),
        )
        self.controller = self._create_controller_instance(scene_name="startup")
        self.action_history =[] # 记录执行过的动作，供 Goal 里的 action_trace 判定 {"action":...,"objectId":...,"pddl_original":...}
        self.init_setup_actions = init_setup_actions
        self._timing_modes = {"abstract", "single_step"} #抽象时间模式, 原始单步模式
        self.set_timing_mode(timing_mode)
        self.teleport_override_path = Path(
            teleport_override_path or Path(__file__).with_name("teleport_overrides.json")
        )
        self.teleport_overrides = {}
        self._load_teleport_overrides()

        # 抽象时间成本（step）配置：静态成本 + 可选动态估算器
        self.action_step_costs = {
            "observe": 4,
            "enter_room": 10,
            "navigate_to": 2,
            "pickup": 4,
            "put_in": 4,
            "open": 6,
            "close": 6,
            "toggle_off": 4,
            "toggle_on": 4,
            "clean": 8,
            "cook": 1,
        }
        if action_step_costs:
            self.action_step_costs.update(action_step_costs)

        self.action_step_estimators: dict[str, Callable] = {
            "navigate_to": self._estimate_navigate_step_cost,
        }
        if action_step_estimators:
            self.action_step_estimators.update(action_step_estimators)

        # ToolRegistry uses this flag to switch from legacy one-shot execution
        # to low-level interleaved scheduling across multiple robots.
        self.supports_low_level_scheduler = True

    def _is_controller_timeout_error(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        message = str(exc).lower()
        return "timed out" in message or "timeout" in message

    def _create_controller_instance(self, scene_name: str) -> Controller:
        last_error: Exception | None = None
        max_attempts = self.controller_reset_timeout_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                with self._controller_reset_timeout(scene_name, operation_name="startup"):
                    return Controller()
            except Exception as exc:
                last_error = exc
                if not self._is_controller_timeout_error(exc) or attempt >= max_attempts:
                    break
                print(
                    f"⚠️ [Env Controller Startup Retry] Timeout on scene={scene_name}, "
                    f"attempt {attempt}/{max_attempts}: {exc}"
                )
                if self.controller_timeout_restart_backoff_sec > 0:
                    time.sleep(self.controller_timeout_restart_backoff_sec)

        raise RuntimeError(
            "Failed to start AI2-THOR controller after timeout retries: "
            f"scene={scene_name}, timeout_retries={self.controller_reset_timeout_retries}"
        ) from last_error

    def _restart_controller_instance(self) -> None:
        # 先停止旧 controller，避免残留后端进程占用资源。
        try:
            if getattr(self, "controller", None) is not None:
                self.controller.stop()
        except Exception:
            pass
        self.controller = None
        self.controller = self._create_controller_instance(scene_name=self.scene_name or "restart")

    @contextmanager
    def _controller_reset_timeout(self, scene_name: str, operation_name: str = "reset"):
        timeout_sec = self.controller_reset_timeout_sec
        if timeout_sec <= 0:
            yield
            return
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        if not hasattr(signal, "SIGALRM"):
            yield
            return

        def _handle_timeout(signum, frame):
            raise TimeoutError(
                f"AI2-THOR controller {operation_name} timed out after {timeout_sec:.1f}s for scene={scene_name}"
            )

        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer: tuple[float, float] | None = None
        signal.signal(signal.SIGALRM, _handle_timeout)
        try:
            if hasattr(signal, "setitimer") and hasattr(signal, "ITIMER_REAL"):
                previous_timer = signal.setitimer(signal.ITIMER_REAL, timeout_sec)
            else:
                signal.alarm(max(1, int(math.ceil(timeout_sec))))
            yield
        finally:
            if hasattr(signal, "setitimer") and hasattr(signal, "ITIMER_REAL"):
                signal.setitimer(signal.ITIMER_REAL, 0)
                if previous_timer and (previous_timer[0] > 0 or previous_timer[1] > 0):
                    signal.setitimer(signal.ITIMER_REAL, *previous_timer)
            else:
                signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    def reset(self, scene_name, agentcount=1):
        """
        重置场景，并执行上帝视角的初始化动作
        ⚠️ TODO. constants 中可以设置初始化参数, 或者,传入某些参数
        """
        self.scene_name = scene_name
        self._load_teleport_overrides()

        last_error: Exception | None = None
        timeout_attempts = 0
        for attempt in range(1, self.controller_reset_retries + 1):
            try:
                with self._controller_reset_timeout(scene_name, operation_name="reset"):
                    self.controller.reset(
                        agentMode="default",
                        massThreshold=None,
                        scene=scene_name,
                        SnaptoGrid = False,
                        visibilityDistance=constants.HORIZON_GRANULARITY,
                        gridSize=constants.AGENT_STEP_SIZE,
                        renderDepthImage=False,
                        renderInstanceSegmentation=False,
                        width=constants.WIDTH, #800,  # /2
                        height=constants.HEIGHT,# 450, # /2
                        fieldOfView=100,
                        agentCount = agentcount,
                    )

                    self.action_history = []

                    print(f"🌍 [Env] 正在布置场景 {scene_name}... (attempt {attempt}/{self.controller_reset_retries})")
                    if self.init_setup_actions:
                        used_slice_names: set[str] = set()
                        for act in self.init_setup_actions:
                            # init_setup_actions 里的都是特权动作，直接执行
                            executable_act = dict(act)
                            resolved_slice_name = ""
                            if executable_act.get("action") == "SliceObject" and executable_act.get("objectTypeHint"):
                                resolved_object_id = self._resolve_init_slice_object_id(
                                    object_type=str(executable_act.get("objectTypeHint") or ""),
                                    used_names=used_slice_names,
                                )
                                if resolved_object_id:
                                    executable_act["objectId"] = resolved_object_id
                                    resolved_slice_name = self._object_name_for_id(resolved_object_id)
                            self.controller.step(**executable_act)
                            if not self.controller.last_event.metadata["lastActionSuccess"]:
                                print(f"  ⚠️ [Env Init Warning] 动作失败: {executable_act} -> {self.controller.last_event.metadata['errorMessage']}")
                            elif executable_act.get("action") == "SliceObject":
                                source_name = resolved_slice_name or self._object_name_for_id(str(executable_act.get("objectId") or ""))
                                if source_name:
                                    used_slice_names.add(source_name)

                return self.get_observation()

            except Exception as exc:
                last_error = exc
                err_type = "Timeout" if self._is_controller_timeout_error(exc) else "ControllerError"
                if err_type == "Timeout":
                    timeout_attempts += 1
                    current_attempt = timeout_attempts
                    max_attempts = self.controller_reset_timeout_retries + 1
                else:
                    current_attempt = attempt
                    max_attempts = self.controller_reset_retries
                print(
                    f"⚠️ [Env Reset Retry] {err_type} on scene={scene_name}, "
                    f"attempt {current_attempt}/{max_attempts}: {exc}"
                )

                if err_type == "Timeout":
                    if timeout_attempts > self.controller_reset_timeout_retries:
                        break
                elif attempt >= self.controller_reset_retries:
                    break

                # 失败后重建 controller，再重试 reset。
                try:
                    self._restart_controller_instance()
                except Exception as restart_exc:
                    last_error = restart_exc
                    print(
                        "⚠️ [Env Reset Retry] controller restart failed "
                        f"on attempt {current_attempt}/{max_attempts}: {restart_exc}"
                    )

                restart_backoff_sec = (
                    self.controller_timeout_restart_backoff_sec
                    if err_type == "Timeout"
                    else self.controller_restart_backoff_sec
                )
                if restart_backoff_sec > 0:
                    time.sleep(restart_backoff_sec)

        raise RuntimeError(
            "Failed to reset AI2-THOR controller after retries: "
            f"scene={scene_name}, retries={self.controller_reset_retries}, "
            f"timeout_retries={self.controller_reset_timeout_retries}"
        ) from last_error

    def _object_name_for_id(self, object_id: str) -> str:
        for obj in self.controller.last_event.metadata.get("objects", []) or []:
            if isinstance(obj, dict) and obj.get("objectId") == object_id:
                return str(obj.get("name") or "")
        return ""

    def _resolve_init_slice_object_id(self, *, object_type: str, used_names: set[str]) -> str | None:
        candidates = [
            obj
            for obj in self.controller.last_event.metadata.get("objects", []) or []
            if isinstance(obj, dict)
            and obj.get("objectType") == object_type
            and obj.get("sliceable")
            and str(obj.get("name") or "") not in used_names
            and obj.get("objectId")
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda obj: str(obj.get("objectId") or ""))
        return str(candidates[0]["objectId"])

    def get_scene_name(self, agent_id: int | None = None) -> str:
        return self.scene_name

    def append_semantic_action_history(
        self,
        *,
        agent_id: int,
        semantic_action: dict,
        success: bool,
        error_msg: str = "",
        planned_steps: int | None = None,
        done_steps: int | None = None,
    ) -> None:
        logged_action = dict(semantic_action)
        logged_action["step_success"] = success
        if planned_steps is not None:
            logged_action["planned_steps"] = int(planned_steps)
        if done_steps is not None:
            logged_action["done_steps"] = int(done_steps)
        if error_msg:
            logged_action["step_error"] = error_msg
        self.action_history.append(logged_action)

    def step(self, semantic_action, agent_id: int = 0):
        """
        执行 Agent 传来的带后缀动作，每次消耗 1 个 step。

        Parameters
        ----------
        semantic_action : dict
            动作描述字典，例如 {"action": "NavigateTo_thor", "objectId": "..."}。
        agent_id : int
            AI2-THOR agentId，多机器人场景下指定操作哪个 robot，默认 0。
        """
        thor_act_name = semantic_action["action"]
        obj_id = semantic_action.get("objectId") or semantic_action.get("object")
        holding_obj_id = semantic_action.get("holdingObjectId")
        room_id = semantic_action.get("room_id") or semantic_action.get("roomId")

        planned_steps = 1
        done_steps = 0
        navigate_estimated_steps = None
        base_steps_consumed = 0
        observation = self.get_observation(agent_id=agent_id)
        
        # ==========================================
        # 核心逻辑：解析后缀并转为 THOR 动作
        # ==========================================
        success = False
        error_msg = ""

        try:
            # ==========================================
            # EmbodiedCoopAgent 语义动作：8 个 Primitive，直接按 action 名分发
            # ==========================================
            if thor_act_name == "observe":
                success, error_msg, observation, base_steps_consumed = self._observe_surroundings(agent_id=agent_id)

            elif thor_act_name == "enter_room":
                if not room_id:
                    success = False
                    error_msg = "Missing required argument: room_id"
                else:
                    scene_name = self._room_id_to_scene_name(room_id)
                    if scene_name == self.scene_name:
                        # 场景已在评测前完成布置，直接复用当前观测，避免重复 reset。
                        success = True
                        error_msg = ""
                    else:
                        last_event = self.controller.last_event
                        events = getattr(last_event, "events", None) or []
                        agent_count = len(events) if events else 1
                        self.reset(scene_name=scene_name, agentcount=agent_count)
                    success = True
                    error_msg = ""
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name == "navigate_to":
                success, error_msg, navigate_estimated_steps, step_consumed = self._teleport_near_object(
                    obj_id,
                    agent_id=agent_id,
                )
                base_steps_consumed = 1 if step_consumed else 0
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name == "pickup":
                success, error_msg = self._execute_thor_action(
                    "PickupObject", agent_id=agent_id, objectId=obj_id, forceAction=True
                )
                base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name == "put_in":
                held_object_id = holding_obj_id or self._get_holding_object_id(agent_id)
                if self._should_place_object_at_origin(held_object_id):
                    success, error_msg = self._place_object_at_point(held_object_id, agent_id=agent_id)
                    if (not success) and obj_id:
                        fallback_success, fallback_error = self._execute_thor_action(
                            "PutObject",
                            agent_id=agent_id,
                            objectId=obj_id,
                            forceAction=True,
                        )
                        if fallback_success:
                            success, error_msg = True, ""
                        else:
                            error_msg = (
                                f"{error_msg}; fallback PutObject failed: {fallback_error}"
                                if error_msg else f"fallback PutObject failed: {fallback_error}"
                            )
                else:
                    success, error_msg = self._execute_thor_action(
                        "PutObject", agent_id=agent_id, objectId=obj_id, forceAction=True
                    )
                base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name == "open":
                success, error_msg = self._execute_thor_action(
                    "OpenObject", agent_id=agent_id, objectId=obj_id
                )
                base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name == "close":
                success, error_msg = self._execute_thor_action(
                    "CloseObject", agent_id=agent_id, objectId=obj_id
                )
                base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name == "toggle_off":
                success, error_msg = self._execute_thor_action(
                    "ToggleObjectOff", agent_id=agent_id, objectId=obj_id
                )
                base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name == "toggle_on":
                success, error_msg = self._execute_thor_action(
                    "ToggleObjectOn", agent_id=agent_id, objectId=obj_id
                )
                base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name == "clean":
                # 逻辑清洁：如果提供了 toolObjectId，必须正在持有该工具
                tool_id = semantic_action.get("toolObjectId")
                if tool_id:
                    inventory = self.controller.last_event.events[agent_id].metadata["inventoryObjects"]
                    holding_ids = [obj["objectId"] for obj in inventory]
                    if tool_id in holding_ids:
                        success = True
                        error_msg = ""
                    else:
                        success = False
                        error_msg = (
                            f"Logical Error: Robot {agent_id} is not holding "
                            f"the required tool {tool_id} (holding: {holding_ids})"
                        )
                else:
                    # 无工具要求：直接记录清洁动作（依赖 history 判定）
                    success = True
                    error_msg = ""
                observation = self.get_agent_observation(agent_id)

            # ==========================================
            # 旧版 BaseAgent 后缀式动作（保持向后兼容）
            # ==========================================
            elif thor_act_name.endswith("_dirty_faucet_open"):
                success, error_msg = self._execute_thor_action("ToggleObjectOff", agent_id=agent_id, objectId=obj_id)
                base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name.endswith("_clean_with_tool_thor"):
                # 处理马桶：假装执行了清理，因为没有物理属性，我们主要依赖 history 记录
                # 约束：执行这个动作时，机器人手里必须拿着规定的工具 (马桶刷)
                inventory = self.controller.last_event.events[agent_id].metadata["inventoryObjects"]
                holding_ids = [obj["objectId"] for obj in inventory]
                tool_id = semantic_action.get("toolObjectId")
                if tool_id and tool_id in holding_ids:
                    success = True
                    error_msg = ""
                else:
                    success = False
                    error_msg = f"Logical Error: Robot is not holding the required tool {tool_id} (Currently holding: {holding_ids})"
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name.endswith("_put_tool"):
                # 扔回地板：模拟器里的 PutObject 不带 receptacle (PlaceObjectAtPoint)
                success, error_msg = self._place_object_at_point(holding_obj_id, agent_id=agent_id)
                if (not success) and obj_id:
                    fallback_success, fallback_error = self._execute_thor_action(
                        "PutObject",
                        agent_id=agent_id,
                        objectId=obj_id,
                        forceAction=True,
                    )
                    if fallback_success:
                        success, error_msg = True, ""
                    else:
                        error_msg = (
                            f"{error_msg}; fallback PutObject failed: {fallback_error}"
                            if error_msg else f"fallback PutObject failed: {fallback_error}"
                        )
                base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)

            elif thor_act_name.endswith("_thor"):
                # 去掉后缀，恢复原生动作名 (e.g., OpenObject_thor -> OpenObject)
                base_act = thor_act_name.replace("_thor", "")

                # NavigateTo 的特殊处理 (Teleport 到物体附近)
                if base_act == "NavigateTo":
                    success, error_msg, navigate_estimated_steps, step_consumed = self._teleport_near_object(
                        obj_id,
                        agent_id=agent_id,
                    )
                    base_steps_consumed = 1 if step_consumed else 0
                elif base_act == "PutObject":
                    success, error_msg = self._execute_thor_action("PutObject",
                                                                   agent_id=agent_id,
                                                                   objectId=obj_id,
                                                                   forceAction=True)
                    base_steps_consumed = 1
                elif base_act == "PickupObject":
                    success, error_msg = self._execute_thor_action(base_act,
                                                                   agent_id=agent_id,
                                                                   objectId=obj_id,
                                                                   forceAction=True)
                    base_steps_consumed = 1
                elif base_act == "OpenObject" and "ShowerCurtain" in obj_id:
                    success, error_msg = self._execute_thor_action(base_act,
                                                                   agent_id=agent_id,
                                                                   objectId=obj_id,
                                                                   openness = 0.0,
                                                                   forceAction=True)
                    base_steps_consumed = 1
                else:
                    success, error_msg = self._execute_thor_action(base_act, 
                                                                   agent_id=agent_id, 
                                                                   objectId=obj_id,
                                                                   forceAction=True)
                    base_steps_consumed = 1
                observation = self.get_agent_observation(agent_id)
            else:
                print(f"❌ [Env Error] 未知的动作格式: {thor_act_name}")
                error_msg = f"Unknown action format: {thor_act_name}"

        except Exception as e:
            success = False
            error_msg = f"Python Exception: {str(e)}"
            observation = self.get_agent_observation(agent_id)

        if self.timing_mode == "single_step":
            planned_steps = 1
        else:
            planned_steps = self._resolve_action_step_cost(
                action_name=thor_act_name,
                semantic_action=semantic_action,
                agent_id=agent_id,
                success=success,
                estimated_steps=navigate_estimated_steps,
            )

        done_steps = max(planned_steps - int(base_steps_consumed), 0)
        if done_steps > 0:
            done_ok, done_error = self._apply_done_steps(done_steps, agent_id=agent_id)
            if not done_ok:
                success = False
                error_msg = f"{error_msg}; Done failed: {done_error}" if error_msg else f"Done failed: {done_error}"

        logged_action = dict(semantic_action)
        logged_action["planned_steps"] = planned_steps
        logged_action["done_steps"] = done_steps
        logged_action["step_success"] = success
        if error_msg:
            logged_action["step_error"] = error_msg
        self.action_history.append(logged_action)

        return observation, success, error_msg

    def _observe_surroundings(self, agent_id: int = 0) -> tuple[bool, str, dict, int]:
        """Rotate around the agent, capture any successful views, and return a merged observation."""
        orientation_order = [
            ("front", "Front View"),
            ("left", "Left View"),
            ("back", "Back View"),
            ("right", "Right View"),
        ]

        captured: list[tuple[str, str, dict]] = [
            ("front", "Front View", self.get_agent_observation(agent_id))
        ]
        successful_rotations = 0
        warnings: list[str] = []

        for _ in range(3):
            success, rotate_error = self._execute_thor_action("RotateLeft", agent_id=agent_id, degrees=90)
            if not success:
                warnings.append(rotate_error or "RotateLeft failed during observe")
                continue

            successful_rotations += 1
            direction_key, direction_label = orientation_order[successful_rotations % 4]
            captured.append((direction_key, direction_label, self.get_agent_observation(agent_id)))

        restore_rotations = (4 - (successful_rotations % 4)) % 4
        successful_restore_rotations = 0
        for _ in range(restore_rotations):
            success, rotate_error = self._execute_thor_action("RotateLeft", agent_id=agent_id, degrees=90)
            if not success:
                warnings.append(rotate_error or "Failed to restore orientation after observe")
                break
            successful_restore_rotations += 1

        ordered_rank = {"front": 0, "left": 1, "right": 2, "back": 3}
        captured.sort(key=lambda item: ordered_rank.get(item[0], 99))

        merged_metadata = self._merge_observation_metadata([obs.get("metadata") for _, _, obs in captured])
        if warnings:
            merged_metadata["observeWarnings"] = warnings

        return True, "", {
            "image": self._compose_observe_views(
                [obs.get("image") for _, _, obs in captured],
                [label for _, label, _ in captured],
            ),
            "metadata": merged_metadata,
            "history": list(self.action_history),
        }, successful_rotations + successful_restore_rotations

    def _compose_observe_views(self, frames: list, labels: list[str]):
        valid_frames = [frame for frame in frames if frame is not None]
        if not valid_frames:
            return None
        if len(valid_frames) != len(frames):
            frames, labels = zip(*[
                (frame, label)
                for frame, label in zip(frames, labels)
                if frame is not None
            ])
            frames = list(frames)
            labels = list(labels)

        pil_frames = [Image.fromarray(frame) for frame in frames]
        width, height = pil_frames[0].size
        frame_count = len(pil_frames)
        columns = 1 if frame_count == 1 else 2
        rows = math.ceil(frame_count / columns)
        canvas = Image.new("RGB", (width * columns, height * rows), color=(0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        positions = [
            ((index % columns) * width, (index // columns) * height)
            for index in range(frame_count)
        ]

        for image, label, (x_pos, y_pos) in zip(pil_frames, labels, positions):
            canvas.paste(image, (x_pos, y_pos))
            draw.text(
                (x_pos + 8, y_pos + 8),
                label,
                fill=(18, 42, 102),
            )

        return np.array(canvas)

    def _merge_observation_metadata(self, metadata_list: list[dict | None]) -> dict:
        merged_metadata: dict = {}
        merged_objects: dict[str, dict] = {}

        for metadata in metadata_list:
            if not isinstance(metadata, dict):
                continue
            if not merged_metadata:
                merged_metadata = dict(metadata)

            for obj in metadata.get("objects") or []:
                if not isinstance(obj, dict):
                    continue
                object_id = obj.get("objectId")
                if not object_id:
                    continue

                existing = merged_objects.get(object_id)
                if existing is None:
                    merged_objects[object_id] = dict(obj)
                    continue

                if obj.get("visible", False):
                    merged_objects[object_id] = dict(obj)
                else:
                    existing["visible"] = bool(existing.get("visible", False) or obj.get("visible", False))

        merged_metadata["objects"] = list(merged_objects.values())
        return merged_metadata

    def set_timing_mode(self, timing_mode: str) -> None:
        """
        设置动作时间模式：
          - abstract: 根据 action 成本补 Done
          - single_step: 每个 action 总计 1 step（兼容原始行为）
        """
        mode = str(timing_mode).strip().lower()
        if mode not in self._timing_modes:
            raise ValueError(
                f"Unsupported timing_mode={timing_mode!r}, expected one of {sorted(self._timing_modes)}"
            )
        self.timing_mode = mode

    def set_action_step_cost(self, action_name: str, step_cost: int) -> None:
        """设置某个动作的静态时间成本（最小为 1）。"""
        self.action_step_costs[action_name] = max(int(step_cost), 1)

    def register_action_step_estimator(self, action_name: str, estimator: Callable) -> None:
        """注册某个动作的动态时间估算器，用于覆盖静态 step 成本。"""
        self.action_step_estimators[action_name] = estimator

    def _resolve_action_step_cost(
        self,
        action_name: str,
        semantic_action: dict,
        agent_id: int,
        success: bool,
        estimated_steps: int | None = None,
    ) -> int:
        """计算动作应消耗的抽象步数，支持静态配置与动态估算。"""
        estimator = self.action_step_estimators.get(action_name)
        if estimator is None and action_name.endswith("_thor"):
            base_action = action_name.replace("_thor", "")
            estimator = self.action_step_estimators.get(base_action)

        if estimator is not None:
            try:
                resolved = estimator(
                    semantic_action=semantic_action,
                    agent_id=agent_id,
                    success=success,
                    estimated_steps=estimated_steps,
                )
            except TypeError:
                # 兼容仅接收旧签名的估算器
                resolved = estimator(semantic_action, agent_id)
            return max(int(resolved), 1)

        if action_name in self.action_step_costs:
            return max(int(self.action_step_costs[action_name]), 1)

        if action_name.endswith("_thor"):
            base_action = action_name.replace("_thor", "")
            return max(int(self.action_step_costs.get(base_action, 1)), 1)

        return 1

    def _apply_done_steps(self, step_count: int, agent_id: int = 0) -> tuple[bool, str]:
        """用 Done 补齐抽象时间。"""
        for _ in range(step_count):
            event = self.controller.step(action="Done", agentId=agent_id)
            events = getattr(event, "events", None) or []
            agent_event = event
            if events and 0 <= agent_id < len(events):
                agent_event = events[agent_id]

            if not agent_event.metadata.get("lastActionSuccess", False):
                return False, agent_event.metadata.get("errorMessage", "Done action failed")

        return True, ""

    def _estimate_navigate_step_cost(
        self,
        semantic_action: dict,
        agent_id: int,
        success: bool,
        estimated_steps: int | None = None,
    ) -> int:
        """navigate_to 的默认耗时估算：优先使用路径估算结果，否则回退到静态配置。"""
        if estimated_steps is not None:
            return max(int(estimated_steps), 1)
        return max(int(self.action_step_costs.get("navigate_to", 1)), 1)

    def _room_id_to_scene_name(self, room_id: str) -> str:
        """将 room_id 规范化为 AI2-THOR 场景名（FloorPlanXXX）。"""
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

    def _load_teleport_overrides(self) -> None:
        """加载外部 JSON 的 teleport 覆盖配置。"""
        if not self.teleport_override_path.exists():
            self.teleport_overrides = {}
            return

        with self.teleport_override_path.open("r", encoding="utf-8") as file_obj:
            loaded = json.load(file_obj)

        if not isinstance(loaded, dict):
            raise ValueError(
                f"Teleport override JSON must be an object: {self.teleport_override_path}"
            )

        self.teleport_overrides = {
            scene_name: scene_config
            for scene_name, scene_config in loaded.items()
            if not str(scene_name).startswith("_")
        }

    def _get_teleport_override(self, obj_id: str) -> dict | None:
        """返回当前场景下指定 objectId 的 teleport 覆盖配置。"""
        if not obj_id:
            return None

        scene_overrides = self.teleport_overrides.get(self.scene_name, {})
        override = scene_overrides.get(obj_id)
        if override and not isinstance(override, dict):
            raise ValueError(
                f"Teleport override for {self.scene_name}/{obj_id} must be an object"
            )
        return override

    def _normalize_teleport_pose(self, obj_id: str, pose: dict, agent_id: int = 0):
        """规范化单个 teleport 位姿配置。"""
        if not isinstance(pose, dict):
            return None, f"Teleport override for {obj_id} must be an object"

        position = pose.get("position")
        if not isinstance(position, dict):
            return None, f"Teleport override for {obj_id} is missing a position object"

        if "x" not in position or "z" not in position:
            return None, f"Teleport override for {obj_id} must include position.x and position.z"

        agent_meta = self.controller.last_event.events[agent_id].metadata["agent"]
        agent_position = agent_meta["position"]
        target_x = float(position["x"])
        target_y = float(position.get("y", agent_position["y"]))
        target_z = float(position["z"])

        rotation = pose.get("rotation", {})
        if isinstance(rotation, (int, float)):
            rotation = {"x": 0.0, "y": float(rotation), "z": 0.0}
        elif isinstance(rotation, dict):
            rotation = {
                "x": float(rotation.get("x", 0.0)),
                "y": float(rotation.get("y", 0.0)),
                "z": float(rotation.get("z", 0.0)),
            }
        else:
            return None, f"Teleport override for {obj_id} has an invalid rotation value"

        normalized_pose = {
            "position": {"x": target_x, "y": target_y, "z": target_z},
            "rotation": rotation,
            "horizon": float(pose.get("horizon", 0.0)),
            "standing": bool(pose.get("standing", True)),
            "estimated_steps": max(
                int(
                    pose.get(
                        "estimated_steps",
                        math.ceil(
                            math.dist(
                                (agent_position["x"], agent_position["z"]),
                                (target_x, target_z),
                            ) / constants.AGENT_STEP_SIZE
                        ) or 1,
                    )
                ),
                1,
            ),
        }
        return normalized_pose, ""

    def _get_teleport_override_candidates(self, obj_id: str, override: dict, agent_id: int = 0):
        """构造 teleport 覆盖候选位姿列表：主位姿优先，其次是备用位姿。"""
        candidates = []

        primary_pose, error_msg = self._normalize_teleport_pose(obj_id, override, agent_id=agent_id)
        if primary_pose is None:
            return None, error_msg
        candidates.append(primary_pose)

        fallback_poses = override.get("fallbacks", [])
        if fallback_poses is None:
            fallback_poses = []
        if not isinstance(fallback_poses, list):
            return None, f"Teleport override for {obj_id} has an invalid fallbacks list"

        for fallback_pose in fallback_poses:
            normalized_pose, error_msg = self._normalize_teleport_pose(
                obj_id,
                fallback_pose,
                agent_id=agent_id,
            )
            if normalized_pose is None:
                return None, error_msg
            candidates.append(normalized_pose)

        return candidates, ""

    def _teleport_with_override(self, obj_id: str, override: dict, agent_id: int = 0):
        """执行外部 JSON 指定的 teleport 位姿，支持失败后的备用位姿。"""
        candidates, error_msg = self._get_teleport_override_candidates(
            obj_id,
            override,
            agent_id=agent_id,
        )
        if candidates is None:
            return False, error_msg, None, False

        last_error = ""
        for index, candidate in enumerate(candidates, start=1):
            event = self.controller.step(
                action="TeleportFull",
                agentId=agent_id,
                position=candidate["position"],
                rotation=candidate["rotation"],
                horizon=candidate["horizon"],
                standing=candidate["standing"],
            )
            agent_event = event.events[agent_id]
            success = agent_event.metadata["lastActionSuccess"]
            error_msg = agent_event.metadata.get("errorMessage", "")
            if success:
                fallback_note = "" if index == 1 else f" (fallback #{index - 1})"
                print(
                    f"    🎯 [TeleportOverride] 使用外部配置: {self.scene_name} / {obj_id}{fallback_note}"
                )
                return True, "", candidate["estimated_steps"], True
            last_error = error_msg

        return False, last_error, candidates[0]["estimated_steps"], True

    def _execute_thor_action(self, action_name, agent_id: int = 0, **kwargs):
        """调用原生 THOR action，通过 agentId 指定操作哪个机器人。"""
        object_id = str(kwargs.get("objectId") or "")
        if action_name == "OpenObject":
            kwargs.setdefault("forceAction", True)
            if "ShowerCurtain" in object_id:
                kwargs.setdefault("openness", 0.0)
        if action_name == "CloseObject":
            kwargs.setdefault("forceAction", True)
            # THOR CloseObject does not accept openness for any object type.
            kwargs.pop("openness", None)
        event = self.controller.step(action=action_name, agentId=agent_id, **kwargs)
        # 多 agent 场景下，通过 events[agent_id] 获取对应 agent 的执行结果
        agent_event = event.events[agent_id]
        success = agent_event.metadata["lastActionSuccess"]
        error_msg = agent_event.metadata.get("errorMessage", "")
        return success, error_msg

    @staticmethod
    def _extract_blocking_shower_curtain_id(error_msg: str) -> str | None:
        """从 THOR 错误信息中提取阻挡放置位置的 ShowerCurtain objectId。"""
        if not error_msg:
            return None
        # 典型格式：Spawn area not clear (ShowerCurtain|-00.22|...)
        match = re.search(r"\((ShowerCurtain\|[^\)]+)\)", str(error_msg))
        if not match:
            return None
        return match.group(1)

    def _drop_held_object_to_floor(self, agent_id: int = 0):
        """兜底：将当前手持物体直接放到 Floor。"""
        return self._execute_thor_action(
            "DropHandObject",
            agent_id=agent_id,
            forceAction=True,
        )

    def _place_object_at_point(self, holding_obj_id, agent_id: int = 0):
        origin_scene_metadata_path = f"env/ai2thor/scene_origin/{self.scene_name}.json"
        with open(origin_scene_metadata_path, "r") as f:
            origin_metadata = json.load(f)

        position = {}
        for obj in origin_metadata.get("objects", []):
            if obj.get("objectId") == holding_obj_id:
                position = obj["position"]
                break

        if not position:
            return False, f"Cannot find origin position for {holding_obj_id}"

        event = self.controller.step(
            action="PlaceObjectAtPoint",
            agentId=agent_id,
            objectId=holding_obj_id,
            position=position,
        )
        agent_event = event.events[agent_id]
        success = agent_event.metadata["lastActionSuccess"]
        error_msg = agent_event.metadata.get("errorMessage", "")

        if not success and self._should_place_object_at_origin(holding_obj_id):
            # scrubbrush 回放到原点时，常见失败是 ShowerCurtain 阻挡 spawn area。
            # 若能识别出门帘，先尝试打开门帘并重试原点放置。
            blocking_curtain_id = self._extract_blocking_shower_curtain_id(error_msg)
            retry_error_msg = ""
            if blocking_curtain_id and "spawn area not clear" in str(error_msg).lower():
                self._execute_thor_action(
                    "OpenObject",
                    agent_id=agent_id,
                    objectId=blocking_curtain_id,
                    openness=0.0,
                )
                retry_event = self.controller.step(
                    action="PlaceObjectAtPoint",
                    agentId=agent_id,
                    objectId=holding_obj_id,
                    position=position,
                )
                retry_agent_event = retry_event.events[agent_id]
                retry_success = retry_agent_event.metadata["lastActionSuccess"]
                retry_error_msg = retry_agent_event.metadata.get("errorMessage", "")
                if retry_success:
                    return True, ""

            # 无论是否命中门帘阻挡，只要 scrubbrush 原点放置失败，都尝试直接放地面。
            drop_success, drop_error_msg = self._drop_held_object_to_floor(agent_id=agent_id)
            if drop_success:
                return True, ""

            if retry_error_msg:
                error_msg = (
                    f"{error_msg}; retry after opening {blocking_curtain_id} failed: {retry_error_msg}"
                    if error_msg else retry_error_msg
                )
            if drop_error_msg:
                error_msg = f"{error_msg}; floor fallback failed: {drop_error_msg}"
        return success, error_msg

    @staticmethod
    def _should_place_object_at_origin(holding_obj_id) -> bool:
        if not holding_obj_id:
            return False
        return str(holding_obj_id).split("|", 1)[0].strip().lower() == "scrubbrush"

    def _get_holding_object_id(self, agent_id: int = 0):
        """返回指定 agent 当前手持物体的 objectId；未持有时返回 None。"""
        try:
            inventory = self.controller.last_event.events[agent_id].metadata.get("inventoryObjects", [])
        except Exception:
            return None

        if not inventory:
            return None

        first_item = inventory[0]
        if isinstance(first_item, dict):
            return first_item.get("objectId")
        return None
    
    def _teleport_near_object(self, obj_id, agent_id: int = 0):
        """
        计算物体附近的可达点，并将机器人瞬间移动到该点，调整视角使其能看到目标。

        Args:
            obj_id: 目标物体的 THOR objectId
            agent_id: AI2-THOR agentId，默认 0

        Returns:
            success (bool): 是否成功移动
            error_msg (str): 错误信息
            estimated_steps (int): 抽象时间估算步数
            step_consumed (bool): 是否已在模拟器中消耗了实际一步
        TODO. 猜测～ 可能出现多个 agent 跳转到同一 object 的情况
        """
        try:
            if not obj_id:
                return False, "Missing required argument: object/objectId", None, False

            override = self._get_teleport_override(obj_id)
            if override:
                success, error_msg, estimated_steps, step_consumed = self._teleport_with_override(
                    obj_id,
                    override,
                    agent_id=agent_id,
                )
                if success:
                    return success, error_msg, estimated_steps, step_consumed
                print(
                    f"    ⚠️ [TeleportOverride] 外部配置失败，回退自动导航点: {self.scene_name} / {obj_id} -> {error_msg}"
                )

            # 1. 调用现有的导航点计算算法
            # 这里我们传入 [obj_id] 作为 filter，只计算目标物体的最佳位置
            target_points_dict = compute_best_nav_points_for_all_objects(
                self.controller,
                object_filter=[obj_id]
            )
            # print(f"    🎯 {obj_id} 的候选导航点：{target_points_dict}")
            # 如果没找到目标物体或没有可达点
            if obj_id not in target_points_dict or not target_points_dict[obj_id]:
                return False, f"[Simulator Error] Cannot find reachable navigation points for {obj_id}", None, False
            
            # 2. 获取排在第一位的最佳候选点
            # coord 的格式是 (x_idx, z_idx, rotation_idx, camera_angle)
            best_coord = target_points_dict[obj_id][0]
            x_idx, z_idx, rot_idx, cam_angle = best_coord
            
            # 3. 将栅格索引 (Index) 转换回真实世界坐标 (World Coordinates)
            # 根据提供的代码逻辑，x_idx = round(x / AGENT_STEP_SIZE)
            world_x = x_idx * constants.AGENT_STEP_SIZE
            world_z = z_idx * constants.AGENT_STEP_SIZE
            
            # Agent 的 Y 坐标需要保持不变 (获取当前高度)
            agent_meta = self.controller.last_event.events[agent_id].metadata['agent']
            agent_y = agent_meta['position']['y']
            current_x = agent_meta['position']['x']
            current_z = agent_meta['position']['z']

            # 用欧式距离估算导航抽象步数（最少 1 步）
            travel_distance = math.dist((current_x, current_z), (world_x, world_z))
            estimated_steps = max(math.ceil(travel_distance / constants.AGENT_STEP_SIZE), 1)
            
            # 4. 将离散的朝向索引转为真实的角度 (Degrees)
            # rotation_idx: 0=North, 1=East, 2=South, 3=West
            rotation_mapping = {
                0: 0.0,    # 面向 +Z
                1: 90.0,   # 面向 +X
                2: 180.0,  # 面向 -Z
                3: 270.0   # 面向 -X
            }
            world_rotation = rotation_mapping.get(rot_idx, 0.0)
            
            # 5. 组装 TeleportFull 指令参数
            # TeleportFull 可以一步到位设置位置、朝向和相机仰角
            teleport_args = {
                "action": "TeleportFull",
                "position": {"x": world_x,"y": agent_y,"z": world_z},
                "rotation": {"x": 0.0, "y": world_rotation, "z": 0.0},
                "horizon": float(cam_angle),
                "standing": True
            }
            
            # 6. 执行瞬间移动
            teleport_args["agentId"] = agent_id
            event = self.controller.step(**teleport_args)
            agent_event = event.events[agent_id]
            success = agent_event.metadata["lastActionSuccess"]
            error_msg = agent_event.metadata.get("errorMessage", "")
            step_consumed = True
            
            # 7. 可选：后备机制 (Fallback)
            # 如果最佳点 Teleport 失败（比如刚好站在了碰撞体边缘），尝试候选点
            if not success and len(target_points_dict[obj_id]) > 1:
                for fallback_coord in target_points_dict[obj_id][1:]:
                    fb_x = fallback_coord[0] * constants.AGENT_STEP_SIZE
                    fb_z = fallback_coord[1] * constants.AGENT_STEP_SIZE
                    fb_rot = rotation_mapping.get(fallback_coord[2], 0.0)
                    fb_cam = float(fallback_coord[3])
                    
                    fallback_args = {
                        "action": "TeleportFull",
                        "position":{"x": fb_x, "y": agent_y, "z": fb_z},
                        "rotation": {"x": 0.0, "y": fb_rot, "z": 0.0},
                        "horizon": fb_cam,
                        "standing": True
                    }
                    fallback_args["agentId"] = agent_id
                    fb_event = self.controller.step(**fallback_args)
                    if fb_event.events[agent_id].metadata["lastActionSuccess"]:
                        return True, "", estimated_steps, True
            
            return success, error_msg, estimated_steps, step_consumed
            
        except Exception as e:
            return False, f"Exception during teleport calculation: {str(e)}", None, False

    def get_observation(self, agent_id: int = 0):
        """返回给 Agent 的观测（兼容单 agent / 多 agent）。"""
        event = self.controller.last_event
        events = getattr(event, "events", None) or []

        if events:
            safe_agent_id = agent_id if 0 <= agent_id < len(events) else 0
            agent_event = events[safe_agent_id]
            return {
                "image": agent_event.frame,
                "metadata": agent_event.metadata,
                "history": self.action_history,
                "scene_name": self.scene_name,
            }

        return {
            "image": event.frame,
            "metadata": event.metadata,
            "history": self.action_history,
            "scene_name": self.scene_name,
        }

    def get_agent_observation(self, agent_id: int = 0):
        """
        返回指定 robot 的独立观测（多 robot 场景下使用）。
        ToolRegistry.execute() 优先调用此方法，以确保返回 metadata 与执行 agent 对应。
        """
        event = self.controller.last_event
        events = getattr(event, "events", None) or []
        if 0 <= agent_id < len(events):
            agent_event = events[agent_id]
            return {
                "image": agent_event.frame,
                "metadata": agent_event.metadata,
                "history": self.action_history,
                "scene_name": self.scene_name,
            }


        # TODO.回退：索引越界时退回到 agent 0（多 agent）或全局事件（单 agent）。
        if events:
            agent_event = events[0]
            metadata = dict(agent_event.metadata)
            metadata["agentObservationWarning"] = (
                f"agent_id {agent_id} out of range for events size {len(events)}; fallback to agent 0"
            )
            return {
                "image": agent_event.frame,
                "metadata": metadata,
                "history": self.action_history,
                "scene_name": self.scene_name,
            }
        # TODO.
        metadata = dict(event.metadata)
        metadata["agentObservationWarning"] = (
            f"agent_id {agent_id} out of range and no per-agent events found; fallback to global observation"
        )
        return {
            "image": event.frame,
            "metadata": metadata,
            "history": self.action_history,
            "scene_name": self.scene_name,
        }
