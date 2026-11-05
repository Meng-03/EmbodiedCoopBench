# eval/eval_embodiedcoop.py
import argparse
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from agents.embodiedcoop.embodiedcoop import EmbodiedCoopAgent
from agents.embodiedcoop.toolregistry import (
    FACILITY_INSPECTION_ACTIONS,
    RESTAURANT_SERVING_ACTIONS,
    RESTOCKING_ACTIONS,
    SEMANTIC_ACTIONS,
)
from robots.team_config import (
    get_robot_team_config,
    list_robot_team_presets,
)
from eval.setup.task_and_team_setup import TaskAndTeamSetup
from env.ai2thor.symbolic_semantic_action import ThorSymbolicEnv
from env.ai2thor.multi_room_env import MultiRoomThorEnv
from eval.setup.scene_special_config import SceneSpecialConfig
import os
from datetime import datetime
from PIL import Image
from typing import Any


REQUEST_INTERVAL_CACHE_REL_PATH = Path("logs") / "embodiedcoop" / "request_injection_interval_cache.json"
AUTO_MAX_ITERATIONS_FLOOR = 40
ABLATION_BASELINE_ROUND_BUFFER = 6
FULL_EXPERIMENT_CATEGORY = "full"
FULL_EXPERIMENT_NAME = "embodiedcoop"
BASELINE_EXPERIMENT_CATEGORY = "baseline"
BASELINE_EXPERIMENT_NAME = "baseline-viki"
BASELINE_ROBOOS_EXPERIMENT_NAME = "baseline-roboos"
AGENT_MODE_EMBODIEDCOOP = "embodiedcoop"
AGENT_MODE_BASELINE_VIKI = "baseline-viki"
AGENT_MODE_BASELINE_ROBOOS = "baseline-roboos"
DEFAULT_REQUEST_TYPE = "hotel_bathroom_housekeeping"
DEFAULT_ACTIVE_SKILL = "hotel_bathroom_housekeeping"
DEFAULT_LOG_SKILL_NAME = "hotel_housekeeping"
DEFAULT_REQUEST_BUNDLE_FILE_PREFIX = "bundles"
DEFAULT_TASK_JSON = "tasks/instances/hotel_housekeeping.json"
DEFAULT_REQUEST_BUNDLE_DIR = "tasks/bundles/hotel_housekeeping"
DEFAULT_EVAL_BUNDLES_FILE = "tasks/manifest.json"
SCENE_CONFIG_FIELDS: tuple[str, ...] = (
    "task_json",
    "request_bundle_dir",
    "request_bundle_file_prefix",
    "robot_team_preset",
    "log_skill_name",
    "request_type",
    "active_skill",
    "request_delivery_mode",
    "initial_request_count",
    "request_injection_interval_rounds",
    "request_injection_batch_size",
)
LEGACY_FULL_EXPERIMENT_GROUPS: tuple[tuple[str, str], ...] = (
    ("other", "baseline"),
    ("other", "embodiedcoop"),
)
_BASELINE_VIKI_AGENT_CLASS = None
_BASELINE_ROBOOS_AGENT_CLASS = None


def _generate_eval_timeline_plots(run_eval_dir: Path) -> None:
    try:
        from eval.analysis.plot_eval_run_timeline import generate_timeline_artifacts

        artifacts = generate_timeline_artifacts(run_eval_dir)
    except Exception as exc:
        print(
            "[EmbodiedCoop][TimelinePlot][WARN] "
            f"failed_to_generate run_dir={run_eval_dir} error={exc}"
        )
        return

    print(
        "[EmbodiedCoop][TimelinePlot] "
        f"output_dir={artifacts['output_dir']} "
        f"message_tool_timeline={artifacts['message_tool_timeline']}"
    )


def _resolve_effective_request_delivery_mode(
    *,
    agent_mode: str,
    requested_mode: str,
) -> tuple[str, str | None]:
    if agent_mode == AGENT_MODE_BASELINE_VIKI and requested_mode != "sequential":
        return "sequential", "baseline-viki receives bundle requests sequentially"
    return requested_mode, None


def _load_baseline_viki_agent_class():
    global _BASELINE_VIKI_AGENT_CLASS
    if _BASELINE_VIKI_AGENT_CLASS is not None:
        return _BASELINE_VIKI_AGENT_CLASS

    module_path = Path(__file__).resolve().parents[1] / "agents" / "baseline-viki" / "baseline_viki_agent.py"
    spec = importlib.util.spec_from_file_location("agents.baseline_viki_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Baseline-Viki agent from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, "BaselineVikiAgent", None)
    if agent_class is None:
        raise ImportError(f"BaselineVikiAgent not found in {module_path}")

    _BASELINE_VIKI_AGENT_CLASS = agent_class
    return agent_class


def _load_baseline_roboos_agent_class():
    global _BASELINE_ROBOOS_AGENT_CLASS
    if _BASELINE_ROBOOS_AGENT_CLASS is not None:
        return _BASELINE_ROBOOS_AGENT_CLASS

    module_path = Path(__file__).resolve().parents[1] / "agents" / "baseline_roboos" / "baseline_roboos_agent.py"
    spec = importlib.util.spec_from_file_location("agents.baseline_roboos_runtime", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Baseline-RoboOS agent from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent_class = getattr(module, "BaselineRoboOSAgent", None)
    if agent_class is None:
        raise ImportError(f"BaselineRoboOSAgent not found in {module_path}")

    _BASELINE_ROBOOS_AGENT_CLASS = agent_class
    return agent_class


def _resolve_agent_class(agent_mode: str):
    if agent_mode == AGENT_MODE_BASELINE_VIKI:
        return _load_baseline_viki_agent_class()
    if agent_mode == AGENT_MODE_BASELINE_ROBOOS:
        return _load_baseline_roboos_agent_class()
    return EmbodiedCoopAgent


def _parse_positive_int_or_range(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        raise argparse.ArgumentTypeError("expected a positive integer or range like 2-4")

    if text.isdigit():
        value = int(text)
        if value <= 0:
            raise argparse.ArgumentTypeError("expected a positive integer")
        return str(value)

    if "-" not in text:
        raise argparse.ArgumentTypeError("expected a positive integer or range like 2-4")

    low_text, high_text = [part.strip() for part in text.split("-", 1)]
    if not low_text.isdigit() or not high_text.isdigit():
        raise argparse.ArgumentTypeError("expected a positive integer range like 2-4")

    low = int(low_text)
    high = int(high_text)
    if low <= 0 or high <= 0:
        raise argparse.ArgumentTypeError("range values must be positive")
    if low > high:
        raise argparse.ArgumentTypeError("range lower bound must be <= upper bound")
    return f"{low}-{high}"


def _parse_positive_int_or_auto(raw: str | int | None) -> str:
    if isinstance(raw, int):
        if raw <= 0:
            raise argparse.ArgumentTypeError("expected 'auto' or a positive integer")
        return str(raw)

    text = str(raw or "").strip().lower()
    if not text or text == "auto":
        return "auto"
    if text.isdigit() and int(text) > 0:
        return text
    raise argparse.ArgumentTypeError("expected 'auto' or a positive integer")


def _request_interval_cache_path(workspace_dir: Path) -> Path:
    return workspace_dir / REQUEST_INTERVAL_CACHE_REL_PATH


def _load_request_interval_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[EmbodiedCoop][RequestDelivery][WARN] failed_to_read_interval_cache path={cache_path} error={exc}")
        return {}
    return data if isinstance(data, dict) else {}


def _save_request_interval_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _build_request_interval_cache_key(
    *,
    request_bundle_file: Path,
    request_bundle: dict[str, Any],
    bundle_label: str,
) -> str:
    bundle_id = str(request_bundle.get("bundle_id") or "").strip()
    if bundle_id:
        bundle_part = f"bundle_id={bundle_id}"
    else:
        request_ids = request_bundle.get("request_ids")
        if isinstance(request_ids, list) and request_ids:
            normalized_request_ids = ",".join(sorted(str(item) for item in request_ids))
            bundle_part = f"request_ids={normalized_request_ids}"
        else:
            bundle_part = f"bundle_label={bundle_label}"
    return f"{request_bundle_file.resolve()}::{bundle_part}"


def _load_bundle_request_interval_cache_context(
    *,
    workspace_dir: Path,
    request_bundle_file: Path,
    request_bundle: dict[str, Any],
    bundle_label: str,
) -> tuple[Path, str, dict[str, Any], dict[str, Any]]:
    cache_path = _request_interval_cache_path(workspace_dir)
    cache_key = _build_request_interval_cache_key(
        request_bundle_file=request_bundle_file,
        request_bundle=request_bundle,
        bundle_label=bundle_label,
    )
    cache = _load_request_interval_cache(cache_path)
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        entry = {}
    return cache_path, cache_key, cache, entry


def _resolve_bundle_request_injection_interval_rounds(
    raw: str | int | None,
    *,
    workspace_dir: Path,
    request_bundle_file: Path,
    request_bundle: dict[str, Any],
    bundle_label: str,
) -> tuple[str, str]:
    del workspace_dir, request_bundle_file, request_bundle, bundle_label
    if isinstance(raw, int):
        value = max(raw, 1)
        return str(value), str(value)

    text = _parse_positive_int_or_range(str(raw or "1"))
    if "-" not in text:
        value = int(text)
        return str(value), text

    return text, f"{text} (sample_per_injection)"


def _sample_request_injection_interval_sequence(interval_spec: str | int, count: int) -> list[int]:
    count = max(int(count or 0), 0)
    if count <= 0:
        return []

    text = _parse_positive_int_or_range(str(interval_spec or "1"))
    if "-" not in text:
        return [int(text)] * count

    low_text, high_text = text.split("-", 1)
    low = int(low_text)
    high = int(high_text)
    candidates = list(range(low, high + 1))
    if count <= len(candidates):
        return random.sample(candidates, count)
    return [random.choice(candidates) for _ in range(count)]


def _derive_request_injection_interval_sequence(request_injections: list[dict[str, Any]]) -> list[int]:
    sequence: list[int] = []
    previous_after_iteration = 0
    for injection in request_injections:
        try:
            after_iteration = int(injection.get("after_iteration") or 0)
        except (TypeError, ValueError):
            continue
        if after_iteration <= 0:
            continue
        sequence.append(max(after_iteration - previous_after_iteration, 1))
        previous_after_iteration = after_iteration
    return sequence


def _get_bundle_total_pddl_plan_length(
    request_bundle: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
) -> int:
    raw_total = request_bundle.get("bundle_total_pddl_plan_length")
    try:
        total = int(raw_total or 0)
    except (TypeError, ValueError):
        total = 0
    if total > 0:
        return total

    total = 0
    for task in selected_tasks:
        if not isinstance(task, dict):
            continue
        plan = task.get("pddl_plan")
        if isinstance(plan, list):
            total += len(plan)
            continue
        try:
            total += int(task.get("pddl_plan_length", 0) or 0)
        except (TypeError, ValueError):
            continue
    return total


def _get_request_bundle_identity(request_bundle: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    bundle_id = str(request_bundle.get("bundle_id") or "").strip()
    request_ids = tuple(
        sorted(str(item) for item in request_bundle.get("request_ids", []) if str(item).strip())
    )
    return bundle_id, request_ids


def _request_bundles_match(candidate_bundle: dict[str, Any], target_bundle: dict[str, Any]) -> bool:
    candidate_bundle_id, candidate_request_ids = _get_request_bundle_identity(candidate_bundle)
    target_bundle_id, target_request_ids = _get_request_bundle_identity(target_bundle)

    if candidate_bundle_id and target_bundle_id:
        return candidate_bundle_id == target_bundle_id
    if candidate_request_ids and target_request_ids:
        return candidate_request_ids == target_request_ids
    return False


def _has_all_successful_request_evaluations(summary_path: Path) -> bool:
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not isinstance(summary, list) or not summary:
        return False

    return all(isinstance(item, dict) and bool(item.get("success")) for item in summary)


def _count_logged_agent_rounds(agent_rounds_dir: Path) -> int:
    try:
        return sum(
            1
            for child in agent_rounds_dir.iterdir()
            if child.is_dir() and child.name.startswith("round_")
        )
    except OSError:
        return 0


def _iter_baseline_eval_run_roots(
    *,
    workspace_dir: Path,
    model: str,
    skill_name: str,
) -> list[Path]:
    grouped_roots = [
        _build_eval_group_dir(
            workspace_dir,
            model=model,
            experiment_category=FULL_EXPERIMENT_CATEGORY,
            experiment_name=FULL_EXPERIMENT_NAME,
            skill_name=skill_name,
        )
    ]
    for experiment_category, experiment_name in LEGACY_FULL_EXPERIMENT_GROUPS:
        grouped_roots.append(
            _build_eval_group_dir(
                workspace_dir,
                model=model,
                experiment_category=experiment_category,
                experiment_name=experiment_name,
                skill_name=skill_name,
            )
        )
    legacy_root = (
        workspace_dir
        / "logs"
        / "embodiedcoop"
        / "eval_runs"
        / _sanitize_log_component(model, fallback="unknown_model")
        / _sanitize_log_component(skill_name, fallback="unknown_skill")
    )

    roots: list[Path] = []
    seen: set[str] = set()
    for root in (*grouped_roots, legacy_root):
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _load_successful_baseline_completion_rounds(
    *,
    workspace_dir: Path,
    model: str,
    skill_name: str,
    request_bundle: dict[str, Any],
) -> list[int]:
    completion_rounds: list[int] = []

    for run_root in _iter_baseline_eval_run_roots(
        workspace_dir=workspace_dir,
        model=model,
        skill_name=skill_name,
    ):
        if not run_root.exists():
            continue
        try:
            run_dirs = list(run_root.iterdir())
        except OSError:
            continue

        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue

            bundle_path = run_dir / "selected_request_bundle.json"
            summary_path = run_dir / "request_evaluations" / "_summary.json"
            agent_rounds_dir = run_dir / "agent_rounds"
            if not bundle_path.exists() or not summary_path.exists() or not agent_rounds_dir.exists():
                continue

            try:
                candidate_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(candidate_bundle, dict):
                continue
            if not _request_bundles_match(candidate_bundle, request_bundle):
                continue
            if not _has_all_successful_request_evaluations(summary_path):
                continue

            round_count = _count_logged_agent_rounds(agent_rounds_dir)
            if round_count > 0:
                completion_rounds.append(round_count)

    return completion_rounds


def _resolve_agent_max_iterations(
    raw: str | int | None,
    *,
    request_bundle: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
    robot_ids: list[int] | None,
    workspace_dir: Path,
    model: str,
    skill_name: str,
    use_baseline_reference_cap: bool,
) -> tuple[int, str]:
    if isinstance(raw, int):
        value = max(raw, 1)
        return value, str(value)

    text = _parse_positive_int_or_auto(raw)
    if text != "auto":
        value = int(text)
        return value, text

    bundle_total_pddl = _get_bundle_total_pddl_plan_length(request_bundle, selected_tasks)
    robot_count = max(len(robot_ids or []), 1)
    request_count = max(int(request_bundle.get("request_count", len(selected_tasks)) or len(selected_tasks) or 1), 1)
    bundle_distinct_room_count = max(int(request_bundle.get("bundle_distinct_room_count") or 0), 0)
    if bundle_distinct_room_count <= 0:
        bundle_distinct_room_count = len(
            {
                str(task.get("scene_name") or task.get("room_id") or "")
                for task in selected_tasks
                if isinstance(task, dict)
            }
        )
    bundle_distinct_room_count = max(bundle_distinct_room_count, 1)

    # 目前不直接应用 robot 折扣因子；如果需要体现并行能力，可以在后续加上
    effective_parallelism = max(1, min(robot_count, 2))

    # 预算由三部分组成：
    # 1. total pddl × 非全知膨胀系数，反映执行/推理/重试开销
    # 2. 房间切换开销：每个 robot 可以反复进入同个房间 5 次
    # 3. 非 robot tool 调用开销：memory 更新、结束确认、理由推理等额外往返
    plan_budget = math.ceil(bundle_total_pddl * 1.5)
    room_switch_budget = robot_count * 5 * max(bundle_distinct_room_count, 0)
    non_robot_tool_budget = 4 * request_count
    estimated = plan_budget + room_switch_budget + non_robot_tool_budget
    auto_value = max(AUTO_MAX_ITERATIONS_FLOOR, estimated)
    label_parts = [
        f"bundle_total_pddl={bundle_total_pddl}",
        f"bundle_distinct_room_count={bundle_distinct_room_count}",
        f"robot_count={robot_count}",
        f"effective_parallelism={effective_parallelism}",
        f"request_count={request_count}",
        f"plan_budget={plan_budget}",
        f"room_switch_budget={room_switch_budget}",
        f"non_robot_tool_budget={non_robot_tool_budget}",
        f"estimated={estimated}",
    ]

    value = auto_value
    if use_baseline_reference_cap:
        baseline_rounds = _load_successful_baseline_completion_rounds(
            workspace_dir=workspace_dir,
            model=model,
            skill_name=skill_name,
            request_bundle=request_bundle,
        )
        if baseline_rounds:
            baseline_success_max_rounds = max(baseline_rounds)
            baseline_cap = max(
                AUTO_MAX_ITERATIONS_FLOOR,
                baseline_success_max_rounds + ABLATION_BASELINE_ROUND_BUFFER,
            )
            value = min(auto_value, baseline_cap)
            label_parts.extend(
                [
                    f"baseline_success_rounds_max={baseline_success_max_rounds}",
                    f"baseline_buffer={ABLATION_BASELINE_ROUND_BUFFER}",
                    f"baseline_cap={baseline_cap}",
                    f"applied_baseline_cap={'yes' if value < auto_value else 'no'}",
                ]
            )

    return (
        value,
        f"auto({', '.join(label_parts)}) -> {value}",
    )


def _sanitize_log_component(value: str | None, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = fallback
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return sanitized or fallback


def _build_experiment_log_group(
    *,
    agent_mode: str = AGENT_MODE_EMBODIEDCOOP,
    ablate_all_memory: bool,
    ablate_agent_memory: bool = False,
) -> tuple[str, str]:
    if agent_mode == AGENT_MODE_BASELINE_VIKI:
        return BASELINE_EXPERIMENT_CATEGORY, BASELINE_EXPERIMENT_NAME
    if agent_mode == AGENT_MODE_BASELINE_ROBOOS:
        return BASELINE_EXPERIMENT_CATEGORY, BASELINE_ROBOOS_EXPERIMENT_NAME

    if ablate_all_memory:
        return "ablation", "all_memory"
    if ablate_agent_memory:
        return "ablation", "agent_memory"
    return FULL_EXPERIMENT_CATEGORY, FULL_EXPERIMENT_NAME


def _build_eval_group_dir(
    workspace_dir: Path,
    *,
    model: str,
    experiment_category: str,
    experiment_name: str,
    skill_name: str,
) -> Path:
    model_dir = _sanitize_log_component(model, fallback="unknown_model")
    category_dir = _sanitize_log_component(experiment_category, fallback="other")
    experiment_dir = _sanitize_log_component(experiment_name, fallback="baseline")
    skill_dir = _sanitize_log_component(skill_name, fallback="unknown_skill")
    return (
        workspace_dir
        / "logs"
        / "embodiedcoop"
        / "eval_runs"
        / model_dir
        / category_dir
        / experiment_dir
        / skill_dir
    )


def _format_robot_action_sets(robot_action_sets: Any) -> str:
    if not robot_action_sets:
        return "同构模式（所有机器人共享同一动作集）"

    if isinstance(robot_action_sets, dict):
        parts: list[str] = []
        for robot_id in sorted(robot_action_sets):
            actions = robot_action_sets.get(robot_id) or []
            parts.append(f"robot{robot_id}({len(actions)}项): {', '.join(map(str, actions))}")
        return "\n".join(parts) if parts else "异构模式（动作集为空）"

    if isinstance(robot_action_sets, list):
        return f"异构模式（共{len(robot_action_sets)}组动作集）"

    return f"异构模式（动作集类型={type(robot_action_sets).__name__}）"


def _robot_team_mode(robot_action_sets: Any) -> str:
    if not robot_action_sets:
        return "homogeneous"
    if isinstance(robot_action_sets, dict):
        normalized = {
            tuple(sorted(str(action) for action in (actions or [])))
            for actions in robot_action_sets.values()
        }
        return "homogeneous" if len(normalized) <= 1 else "heterogeneous"
    return "heterogeneous"


def _resolve_extra_actions_for_robot_team(
    *,
    active_skill: str,
    robot_action_sets: Any,
) -> list[dict[str, Any]] | None:
    base_action_names = {str(spec.get("name")) for spec in SEMANTIC_ACTIONS if isinstance(spec, dict)}
    requested_actions: set[str] = set()
    if isinstance(robot_action_sets, dict):
        for actions in robot_action_sets.values():
            requested_actions.update(str(action) for action in (actions or []) if str(action))

    extra_actions: list[dict[str, Any]] = []
    normalized_skill = str(active_skill).replace("-", "_")

    scenario_actions_by_skill = {
        "facility_inspection": FACILITY_INSPECTION_ACTIONS,
        "restocking": RESTOCKING_ACTIONS,
        "inventory_replenishment": RESTOCKING_ACTIONS,
        "restaurant_serving": RESTAURANT_SERVING_ACTIONS,
    }
    for skill_name, action_specs in scenario_actions_by_skill.items():
        action_names = {
            str(spec.get("name"))
            for spec in action_specs
            if isinstance(spec, dict)
        }
        needs_actions = (
            normalized_skill == skill_name
            or bool((requested_actions - base_action_names) & action_names)
        )
        if needs_actions:
            for spec in action_specs:
                if isinstance(spec, dict) and spec not in extra_actions:
                    extra_actions.append(spec)

    return extra_actions or None


def _print_eval_run_overview(
    *,
    args: argparse.Namespace,
    agent_mode: str,
    task_json_path: str,
    preset_name: str,
    team_config: dict[str, Any],
    robot_ids: list[int],
    robot_action_sets: Any,
    request_bundle_file: Path,
    request_bundle: dict[str, Any],
    bundle_label: str,
    selected_tasks: list[dict[str, Any]],
    model: str,
    request_type: str,
    active_skill: str,
    max_iterations: int,
    max_iterations_label: str,
    run_eval_dir: Path,
    memory_file: Path,
    interval_rounds_label: str,
    request_injections: list[dict[str, Any]],
    initial_tasks: list[dict[str, Any]],
    effective_request_delivery_mode: str,
    request_delivery_override_reason: str | None,
) -> None:
    print("\n" + "=" * 78)
    print("[EmbodiedCoop][测试总览] 本次测试配置（中文）")
    print("=" * 78)
    print(f"- 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"- agent_mode: {agent_mode}")
    print(f"- 模型: {model}")
    print(f"- 技能标签(log_skill_name): {args.log_skill_name}")
    print(f"- 请求类型(request_type): {request_type}")
    print(f"- 活跃技能(active_skill): {active_skill}")
    print(f"- 任务文件(task_json): {task_json_path}")
    print(f"- 输出目录(run_eval_dir): {run_eval_dir}")
    print(f"- memory文件: {memory_file}")

    print("\n[任务与请求信息]")
    print(f"- bundle来源: {request_bundle_file}")
    print(f"- bundle标识: {bundle_label}")
    print(f"- bundle请求数: {request_bundle.get('request_count')}")
    print(f"- bundle请求ID: {request_bundle.get('request_ids')}")
    print(f"- 实际选中任务数: {len(selected_tasks)}")
    for idx, task in enumerate(selected_tasks, start=1):
        print(
            f"  {idx}. request_id={task.get('request_id')} "
            f"instance_id={task.get('instance_id')} "
            f"scene={task.get('scene_name')} room={task.get('room_id')}"
        )

    print("\n[机器人信息]")
    print(f"- robot_team_preset: {preset_name}")
    print(f"- 队伍描述: {team_config.get('description', '')}")
    print(f"- 机器人ID: {robot_ids}")
    print(f"- 队伍模式: {'异构' if _robot_team_mode(robot_action_sets) == 'heterogeneous' else '同构'}")
    print("- 动作集详情:")
    print(_format_robot_action_sets(robot_action_sets))

    print("\n[执行与注入策略]")
    print(f"- 最大迭代轮数: {max_iterations} (来源: {max_iterations_label})")
    print(f"- 请求投递模式: {effective_request_delivery_mode}")
    if request_delivery_override_reason:
        print(
            f"- 请求投递模式覆盖: {args.request_delivery_mode} -> {effective_request_delivery_mode} "
            f"({request_delivery_override_reason})"
        )
    print(f"- 初始请求数: {len(initial_tasks)}")
    print(f"- 待注入批次数: {len(request_injections)}")
    print(f"- 注入间隔轮数: {interval_rounds_label}")
    print(f"- 每批注入请求数: {args.request_injection_batch_size}")

    print("\n[消融配置]")
    print(f"- ablate_all_memory: {args.ablate_all_memory}")
    print(f"- ablate_agent_memory: {args.ablate_agent_memory}")
    print("=" * 78)


def _default_config_path() -> Path:
    return Path(__file__).resolve().parent / "evalconfig" / "eval_embodiedcoop_config.yaml"


def _load_config_file(config_file: str | None) -> dict[str, Any]:
    path = Path(config_file) if config_file else _default_config_path()
    if not path.exists():
        return {}

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML config files. "
                "Install it in your environment or use a JSON config file instead."
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file suffix: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping, got {type(data).__name__}")
    return data


def _resolve_workspace_path(path_text: str | Path | None) -> Path:
    path = Path(path_text or "")
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return WORKSPACE_DIR / path


def _infer_request_bundle_count(bundle_id: str) -> int | None:
    match = re.match(r"^req(\d+)_", str(bundle_id or ""))
    if not match:
        return None
    return int(match.group(1))


def _load_eval_bundle_index(eval_bundles_file: str | Path | None) -> dict[str, Any]:
    path = _resolve_workspace_path(eval_bundles_file or DEFAULT_EVAL_BUNDLES_FILE)
    if not path.exists():
        raise FileNotFoundError(f"Eval bundles file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scenes = data.get("scenes")
    if not isinstance(scenes, dict):
        raise ValueError(f"Eval bundles file must contain a top-level scenes mapping: {path}")
    return data


def _find_eval_bundle(scene_config: dict[str, Any], bundle_id: str) -> dict[str, Any]:
    for bundle in scene_config.get("bundles", []):
        if isinstance(bundle, dict) and str(bundle.get("bundle_id") or "") == bundle_id:
            return bundle
    raise ValueError(f"Unknown bundle_id '{bundle_id}' for scene")


def _expand_scene_config(config: dict[str, Any]) -> dict[str, Any]:
    scene_name = str(config.get("scene") or "").strip()
    if not scene_name:
        return dict(config)

    eval_bundles_file = config.get("eval_bundles_file") or DEFAULT_EVAL_BUNDLES_FILE
    eval_bundle_index = _load_eval_bundle_index(eval_bundles_file)
    scenes = eval_bundle_index["scenes"]
    if scene_name not in scenes:
        available = ", ".join(sorted(scenes.keys()))
        raise ValueError(f"Unknown eval scene '{scene_name}'. Available scenes: {available}")

    scene_config = scenes[scene_name]
    if not isinstance(scene_config, dict):
        raise ValueError(f"Invalid eval scene config for '{scene_name}'")

    expanded = dict(config)
    expanded["eval_bundles_file"] = str(eval_bundles_file)
    for key in SCENE_CONFIG_FIELDS:
        if key in scene_config and not expanded.get(key):
            expanded[key] = scene_config[key]

    bundle_id = str(expanded.get("bundle_id") or expanded.get("request_bundle_id") or "").strip()
    if bundle_id:
        bundle = _find_eval_bundle(scene_config, bundle_id)
        expanded["request_bundle_id"] = bundle_id
        if not expanded.get("request_bundle_count"):
            expanded["request_bundle_count"] = (
                bundle.get("request_bundle_count")
                or _infer_request_bundle_count(bundle_id)
            )

    return expanded


def _extract_scene_cli_overrides(raw_argv: list[str]) -> dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--bundle-id", type=str, default=None)
    parser.add_argument("--request-bundle-id", type=str, default=None)
    parser.add_argument("--eval-bundles-file", type=str, default=None)
    args, _ = parser.parse_known_args(raw_argv)
    overrides: dict[str, Any] = {}
    for key in ("scene", "bundle_id", "request_bundle_id", "eval_bundles_file"):
        value = getattr(args, key, None)
        if value not in (None, ""):
            overrides[key] = value
    return overrides


def _config_dict_to_argv(config: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in config.items():
        if key == "config_file":
            continue
        arg_name = f"--{key.replace('_', '-') }"
        if isinstance(value, bool):
            if value:
                argv.append(arg_name)
        elif isinstance(value, (list, tuple)):
            argv.append(arg_name)
            argv.append(",".join(str(item) for item in value))
        else:
            argv.append(arg_name)
            argv.append(str(value))
    return argv

def _get_request_id_from_task(task: dict[str, Any], fallback_index: int) -> str:
    return str(task.get("request_id") or f"req_{task.get('instance_id', fallback_index)}")


def _request_type_label(request_type: str | None) -> str:
    normalized = str(request_type or DEFAULT_REQUEST_TYPE).replace("-", "_").strip("_").lower()
    if normalized == DEFAULT_REQUEST_TYPE:
        return "housekeeping"
    return normalized.replace("_", " ")


def _request_line_prefix(request_type: str | None) -> str:
    normalized = str(request_type or DEFAULT_REQUEST_TYPE).replace("-", "_").strip("_").lower()
    if normalized == "facility_inspection":
        return "Inspection request: "
    if normalized == "restocking":
        return "Restocking request: "
    if normalized == "restaurant_serving":
        return "Serving request: "
    return ""


def _build_request_message(
    tasks: list[dict[str, Any]],
    *,
    intro: str | None = None,
    request_type: str | None = None,
) -> str:
    normalized_tasks = [task for task in tasks if isinstance(task, dict)]
    if not normalized_tasks:
        return ""

    if intro is None:
        request_label = _request_type_label(request_type)
        intro = (
            f"You receive multiple {request_label} requests simultaneously:"
            if len(normalized_tasks) > 1
            else f"You receive a {request_label} request:"
        )

    lines = [intro]
    line_prefix = _request_line_prefix(request_type)
    for idx, task in enumerate(normalized_tasks, start=1):
        normalized_request_type = str(request_type or DEFAULT_REQUEST_TYPE).replace("-", "_").strip("_").lower()
        if normalized_request_type == "restaurant_serving":
            order = task.get("order") or {}
            fried_egg_count = int(order.get("fried_egg_count", 0) or 0) if isinstance(order, dict) else 0
            toast_count = int(order.get("toast_count", 0) or 0) if isinstance(order, dict) else 0
            request_id = _get_request_id_from_task(task, idx)
            lines.append(
                f"{idx}. {line_prefix}need {fried_egg_count} fried egg(s), "
                f"need {toast_count} slice(s) of toast. request_id={request_id}"
            )
            continue
        room_id = task.get("room_id") or str(task.get("scene_name", "")).replace("FloorPlan", "")
        request_id = _get_request_id_from_task(task, idx)
        lines.append(f"{idx}. {line_prefix}Room {room_id} request_id={request_id}")
    return "\n".join(lines)


def _build_incremental_request_intro(task_count: int, *, request_type: str | None = None) -> str:
    request_label = _request_type_label(request_type)
    return (
        f"A new {request_label} request arrives while previous requests may still be in progress:"
        if task_count == 1
        else f"New {request_label} requests arrive while previous requests may still be in progress:"
    )


def _build_request_injection_entry(
    *,
    batch: list[dict[str, Any]],
    after_iteration: int,
    request_type: str,
    active_skill: str,
) -> dict[str, Any]:
    return {
        "after_iteration": after_iteration,
        "message": _build_request_message(
            batch,
            intro=_build_incremental_request_intro(len(batch), request_type=request_type),
            request_type=request_type,
        ),
        "tasks": batch,
        "request_type": request_type,
        "active_skill": active_skill,
    }


def _build_request_delivery_plan_snapshot(
    *,
    selected_tasks: list[dict[str, Any]],
    initial_tasks: list[dict[str, Any]],
    request_injections: list[dict[str, Any]],
    interval_spec: str | int,
    initial_request_count: int,
    injection_batch_size: int,
) -> dict[str, Any]:
    request_order = [_get_request_id_from_task(task, idx) for idx, task in enumerate(selected_tasks, start=1)]
    initial_request_ids = [_get_request_id_from_task(task, idx) for idx, task in enumerate(initial_tasks, start=1)]

    per_request_after_iteration: list[dict[str, Any]] = [
        {
            "request_id": request_id,
            "after_iteration": 0,
        }
        for request_id in initial_request_ids
    ]

    injections_snapshot: list[dict[str, Any]] = []
    for injection in request_injections:
        try:
            after_iteration = int(injection.get("after_iteration") or 0)
        except (TypeError, ValueError):
            after_iteration = 0
        batch_tasks = [task for task in injection.get("tasks", []) if isinstance(task, dict)]
        batch_request_ids = [
            _get_request_id_from_task(task, len(per_request_after_iteration) + offset)
            for offset, task in enumerate(batch_tasks, start=1)
        ]
        injections_snapshot.append(
            {
                "after_iteration": after_iteration,
                "request_ids": batch_request_ids,
            }
        )
        for request_id in batch_request_ids:
            per_request_after_iteration.append(
                {
                    "request_id": request_id,
                    "after_iteration": after_iteration,
                }
            )

    return {
        "interval_spec": str(interval_spec),
        "interval_sequence": _derive_request_injection_interval_sequence(request_injections),
        "initial_request_count": int(max(initial_request_count, 1)),
        "injection_batch_size": int(max(injection_batch_size, 1)),
        "request_count": len(selected_tasks),
        "request_order": request_order,
        "initial_request_ids": initial_request_ids,
        "per_request_after_iteration": per_request_after_iteration,
        "injections": injections_snapshot,
    }


def _restore_cached_request_delivery_plan(
    *,
    selected_tasks: list[dict[str, Any]],
    cached_delivery_plan: dict[str, Any] | None,
    expected_interval_spec: str | int,
    request_type: str,
    active_skill: str,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]]] | None:
    if not isinstance(cached_delivery_plan, dict):
        return None

    expected_spec = _parse_positive_int_or_range(str(expected_interval_spec or "1"))
    cached_spec = str(cached_delivery_plan.get("interval_spec") or cached_delivery_plan.get("interval_rounds") or "").strip()
    if not cached_spec:
        return None
    try:
        cached_spec = _parse_positive_int_or_range(cached_spec)
    except argparse.ArgumentTypeError:
        return None
    if cached_spec != expected_spec:
        return None

    initial_request_ids_raw = cached_delivery_plan.get("initial_request_ids")
    injections_raw = cached_delivery_plan.get("injections")
    if not isinstance(initial_request_ids_raw, list) or not isinstance(injections_raw, list):
        return None

    tasks_by_request_id: dict[str, dict[str, Any]] = {}
    for idx, task in enumerate(selected_tasks, start=1):
        request_id = _get_request_id_from_task(task, idx)
        if request_id in tasks_by_request_id:
            return None
        tasks_by_request_id[request_id] = task

    initial_request_ids = [str(item) for item in initial_request_ids_raw if str(item)]
    if not initial_request_ids:
        return None

    try:
        initial_tasks = [tasks_by_request_id[request_id] for request_id in initial_request_ids]
    except KeyError:
        return None

    seen_request_ids = list(initial_request_ids)
    request_injections: list[dict[str, Any]] = []
    for injection in injections_raw:
        if not isinstance(injection, dict):
            return None

        try:
            after_iteration = int(injection.get("after_iteration") or 0)
        except (TypeError, ValueError):
            return None
        if after_iteration <= 0:
            return None

        request_ids_raw = injection.get("request_ids")
        if not isinstance(request_ids_raw, list) or not request_ids_raw:
            return None
        batch_request_ids = [str(item) for item in request_ids_raw if str(item)]
        try:
            batch_tasks = [tasks_by_request_id[request_id] for request_id in batch_request_ids]
        except KeyError:
            return None

        seen_request_ids.extend(batch_request_ids)
        request_injections.append(
            _build_request_injection_entry(
                batch=batch_tasks,
                after_iteration=after_iteration,
                request_type=request_type,
                active_skill=active_skill,
            )
        )

    if len(seen_request_ids) != len(selected_tasks):
        return None
    if len(set(seen_request_ids)) != len(seen_request_ids):
        return None
    if set(seen_request_ids) != set(tasks_by_request_id):
        return None

    return initial_tasks, _build_request_message(initial_tasks, request_type=request_type), request_injections


def _persist_bundle_request_delivery_plan(
    *,
    workspace_dir: Path,
    request_bundle_file: Path,
    request_bundle: dict[str, Any],
    bundle_label: str,
    selected_tasks: list[dict[str, Any]],
    initial_tasks: list[dict[str, Any]],
    request_injections: list[dict[str, Any]],
    interval_spec: str | int,
    initial_request_count: int,
    injection_batch_size: int,
    source_spec: str | int | None,
) -> None:
    cache_path, cache_key, cache, entry = _load_bundle_request_interval_cache_context(
        workspace_dir=workspace_dir,
        request_bundle_file=request_bundle_file,
        request_bundle=request_bundle,
        bundle_label=bundle_label,
    )

    updated_entry = dict(entry)
    interval_sequence = _derive_request_injection_interval_sequence(request_injections)
    updated_entry.update(
        {
            "interval_spec": str(interval_spec),
            "interval_sequence": interval_sequence,
            "source_spec": str(source_spec or interval_spec),
            "bundle_id": str(request_bundle.get("bundle_id") or ""),
            "bundle_label": bundle_label,
            "request_count": int(request_bundle.get("request_count") or len(selected_tasks)),
            "request_ids": [str(item) for item in request_bundle.get("request_ids", []) if str(item)],
            "request_bundle_file": str(request_bundle_file.resolve()),
            "request_delivery_plan": _build_request_delivery_plan_snapshot(
                selected_tasks=selected_tasks,
                initial_tasks=initial_tasks,
                request_injections=request_injections,
                interval_spec=interval_spec,
                initial_request_count=initial_request_count,
                injection_batch_size=injection_batch_size,
            ),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    cache[cache_key] = updated_entry
    _save_request_interval_cache(cache_path, cache)


def _build_request_delivery_plan(
    *,
    selected_tasks: list[dict[str, Any]],
    request_bundle: dict[str, Any],
    delivery_mode: str,
    initial_request_count: int,
    injection_interval_rounds: str | int,
    injection_batch_size: int,
    request_type: str,
    active_skill: str,
    cached_delivery_plan: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]], str]:
    normalized_tasks = [task for task in selected_tasks if isinstance(task, dict)]
    if not normalized_tasks:
        return [], "", [], "empty"

    if delivery_mode != "incremental" or len(normalized_tasks) == 1:
        return (
            normalized_tasks,
            request_bundle.get("combined_request_text") or _build_request_message(normalized_tasks, request_type=request_type),
            [],
            "direct_delivery",
        )

    restored_plan = _restore_cached_request_delivery_plan(
        selected_tasks=normalized_tasks,
        cached_delivery_plan=cached_delivery_plan,
        expected_interval_spec=injection_interval_rounds,
        request_type=request_type,
        active_skill=active_skill,
    )
    if restored_plan is not None:
        initial_tasks, initial_message, request_injections = restored_plan
        return initial_tasks, initial_message, request_injections, "cached_bundle_schedule"

    initial_count = min(max(int(initial_request_count or 1), 1), len(normalized_tasks))
    batch_size = max(int(injection_batch_size or 1), 1)

    initial_tasks = normalized_tasks[:initial_count]
    pending_tasks = normalized_tasks[initial_count:]
    initial_message = _build_request_message(initial_tasks, request_type=request_type)
    injection_count = math.ceil(len(pending_tasks) / batch_size) if pending_tasks else 0
    interval_sequence = _sample_request_injection_interval_sequence(injection_interval_rounds, injection_count)

    request_injections: list[dict[str, Any]] = []
    next_iteration = 0
    for interval_rounds, start in zip(interval_sequence, range(0, len(pending_tasks), batch_size)):
        batch = pending_tasks[start:start + batch_size]
        next_iteration += interval_rounds
        request_injections.append(
            _build_request_injection_entry(
                batch=batch,
                after_iteration=next_iteration,
                request_type=request_type,
                active_skill=active_skill,
            )
        )

    return initial_tasks, initial_message, request_injections, "generated_bundle_schedule"

def _normalize_goal_object_type(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split("|", 1)[0].strip().lower()


def _extract_goal_actual_and_expected_value(
    curr_obj: dict[str, Any],
    attr: str,
    target_val: Any,
) -> tuple[Any, Any]:
    compare_target = target_val
    normalized_type = _normalize_goal_object_type(
        curr_obj.get("objectType") or curr_obj.get("objectId")
    )
    if (
        attr == "isOpen"
        and normalized_type == "showercurtain"
        and curr_obj.get("openness") is not None
    ):
        actual_val = float(curr_obj.get("openness")) < 0.5
    elif (
        attr == "openness"
        and normalized_type == "showercurtain"
        and isinstance(target_val, (int, float))
    ):
        actual_val = float(curr_obj.get("openness"))
        compare_target = 1.0 - float(target_val)
    else:
        actual_val = curr_obj.get(attr)
    return actual_val, compare_target


def _evaluate_count_by_type_parent_goal(
    *,
    specs: Any,
    metadata_object_records: list[dict[str, Any]],
    scene_name: str,
    semantic_cooked_object_ids: set[str] | None = None,
    semantic_served_object_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if isinstance(specs, dict):
        count_specs = [specs]
    elif isinstance(specs, list):
        count_specs = [spec for spec in specs if isinstance(spec, dict)]
    else:
        count_specs = []

    object_results: list[dict[str, Any]] = []
    failed_reasons: list[str] = []
    for spec in count_specs:
        object_type = str(spec.get("objectType") or spec.get("object_type") or "").strip()
        target_parent = str(spec.get("parentReceptacleId") or spec.get("parent") or "").strip()
        min_count_mode = "min_count" in spec or "minCount" in spec
        expected_count = int(
            spec.get("min_count", spec.get("minCount", spec.get("count", 0))) or 0
        )
        required_state = spec.get("state") or spec.get("attributes") or {}
        if not isinstance(required_state, dict):
            required_state = {}
        normalized_type = _normalize_goal_object_type(object_type)
        valid_receptacles = SceneSpecialConfig.get_valid_receptacle_ids(scene_name, target_parent)

        matched_object_ids: list[str] = []
        candidate_object_ids: list[str] = []
        semantic_cooked_object_ids = semantic_cooked_object_ids or set()
        semantic_served_object_ids = semantic_served_object_ids or set()
        for obj in metadata_object_records:
            object_id = str(obj.get("objectId") or "")
            actual_type = _normalize_goal_object_type(obj.get("objectType") or obj.get("objectId"))
            type_matches = actual_type == normalized_type
            if (
                not type_matches
                and normalized_type == "breadsliced"
                and actual_type == "bread"
                and object_id in semantic_cooked_object_ids
            ):
                type_matches = True
            if (
                not type_matches
                and normalized_type == "eggcracked"
                and actual_type == "egg"
                and object_id in semantic_cooked_object_ids
            ):
                type_matches = True
            if not type_matches:
                continue
            state_matches = True
            for attr, value in required_state.items():
                actual_value = obj.get(attr)
                if (
                    attr == "isCooked"
                    and value is True
                    and object_id in semantic_cooked_object_ids
                ):
                    actual_value = True
                if actual_value != value:
                    state_matches = False
                    break
            if not state_matches:
                continue
            candidate_object_ids.append(object_id)
            parents = obj.get("parentReceptacles") or []
            if not isinstance(parents, list):
                parents = [parents]
            if (
                valid_receptacles.intersection({str(parent) for parent in parents if parent})
                or object_id in semantic_served_object_ids
            ):
                matched_object_ids.append(str(obj.get("objectId") or ""))

        actual_count = len(matched_object_ids)
        passed = actual_count >= expected_count if min_count_mode else actual_count == expected_count
        message = ""
        if not passed:
            comparator = "at least" if min_count_mode else "count"
            message = (
                f"{object_type} in {sorted(valid_receptacles)} expected {comparator} "
                f"{expected_count}, actual {actual_count}"
            )
            failed_reasons.append(message)

        object_results.append(
            {
                "object_id": f"count:{normalized_type}@{target_parent}",
                "found": True,
                "object_type": object_type,
                "final_object_state": {
                    "matched_object_ids": matched_object_ids,
                    "candidate_object_ids": candidate_object_ids,
                },
                "category_fallback_used": False,
                "checks": [
                    {
                        "attribute": "countByTypeParent",
                        "check_type": "count_by_type_parent",
                        "expected": expected_count,
                        "actual": actual_count,
                        "comparison": "min_count" if min_count_mode else "exact_count",
                        "objectType": object_type,
                        "parentReceptacleId": target_parent,
                        "requiredState": dict(required_state),
                        "valid_receptacles": sorted(valid_receptacles),
                        "passed": passed,
                        "message": message,
                    }
                ],
            }
        )

    return object_results, failed_reasons


def evaluate_goal_detailed(goal_verification, metadata_objects, action_history, scene_name="") -> dict[str, Any]:
    """
    返回: (is_success: bool, failed_reasons: list)
    """
    failed_reasons = []
    metadata_object_records = [
        obj for obj in (metadata_objects or []) if isinstance(obj, dict) and obj.get("objectId")
    ]
    id_to_obj = {
        obj["objectId"]: obj
        for obj in metadata_object_records
    }
    objects_by_type: dict[str, list[dict[str, Any]]] = {}
    for obj in metadata_object_records:
        normalized_type = _normalize_goal_object_type(obj.get("objectType") or obj.get("objectId"))
        if normalized_type:
            objects_by_type.setdefault(normalized_type, []).append(obj)
    action_history_records = [record for record in (action_history or []) if isinstance(record, dict)]
    action_history_act = [act.get("action") for act in action_history_records if act.get("action")]
    semantic_cooked_object_ids = {
        str(record.get("semantic_cooked_objectId"))
        for record in action_history_records
        if record.get("step_success") and record.get("semantic_cooked_objectId")
    }
    semantic_served_object_ids = {
        str(record.get("semantic_served_objectId"))
        for record in action_history_records
        if record.get("step_success") and record.get("semantic_served_objectId")
    }
    object_results: list[dict[str, Any]] = []

    for obj_id, checks in (goal_verification or {}).items():
        if obj_id == "__count_by_type_parent__":
            count_results, count_failed = _evaluate_count_by_type_parent_goal(
                specs=checks,
                metadata_object_records=metadata_object_records,
                scene_name=scene_name,
                semantic_cooked_object_ids=semantic_cooked_object_ids,
                semantic_served_object_ids=semantic_served_object_ids,
            )
            object_results.extend(count_results)
            failed_reasons.extend(count_failed)
            continue

        # 0. 机器人相关的独立检查（可以放在外层，比如看 inventory 是否为空
        if obj_id == "robot1":
            continue

        object_result: dict[str, Any] = {
            "object_id": obj_id,
            "found": False,
            "object_type": None,
            "final_object_state": None,
            "category_fallback_used": False,
            "checks": [],
        }
        normalized_goal_type = _normalize_goal_object_type(obj_id)
        curr_obj = id_to_obj.get(obj_id)
        if not curr_obj and normalized_goal_type == "cabinet":
            fallback_candidates = objects_by_type.get(normalized_goal_type, [])
            if fallback_candidates:
                curr_obj = fallback_candidates[0]
                object_result["category_fallback_used"] = True
        if not curr_obj:
            message = f"Object {obj_id} 不存在于最终场景中"
            failed_reasons.append(message)
            object_result["checks"].append(
                {
                    "attribute": "__object_exists__",
                    "check_type": "existence",
                    "expected": True,
                    "actual": False,
                    "passed": False,
                    "message": message,
                }
            )
            object_results.append(object_result)
            continue

        object_result["found"] = True
        object_result["object_type"] = curr_obj.get("objectType")
        object_result["final_object_state"] = curr_obj
        
        # "Toilet|-00.52|00.00|+03.22": {
        #     "isOpen": false,
        #     "requireActionTrace": "CleanObject_clean_with_tool_thor"
        # }
        for attr, target_val in checks.items():
            # 1. 检查动作轨迹 (如马桶清理)
            # ⚠️ 看一下 action_history 传递过来的是什么形式的
            # TODO. 【需要修改】 这里只检查了 CleanObject 这一个动作
            # TODO. 而且这里 应该是对应 semantic action name [CleanObject Toiletxxx], 
            # TODO. 不是 pddl 中的 action name [CleanObject_clean_with_tool_thor]
            if attr == "requireActionTrace":
                # EmbodiedCoop 记录 'clean'，兼容旧式动作名（如 CleanObject_clean_with_tool_thor）
                _action_aliases = {
                    "CleanObject_clean_with_tool_thor": "clean",
                }
                alias = _action_aliases.get(target_val)
                matched = target_val in action_history_act or (alias is not None and alias in action_history_act)
                message = ""
                if not matched:
                    message = f"{obj_id} 缺失必要的动作轨迹: {target_val}"
                    failed_reasons.append(message)
                object_result["checks"].append(
                    {
                        "attribute": attr,
                        "check_type": "action_trace",
                        "expected": target_val,
                        "alias": alias,
                        "actual": action_history_act,
                        "passed": matched,
                        "message": message,
                    }
                )
            
            # 2. 检查父容器 (parentReceptacleId)
            #    通过 SceneSpecialConfig 扩展有效容器集合，兼容特殊场景映射
            #    例如 402 场景中 Sink 映射到 CounterTop，两者都算合法
            elif attr == "parentReceptacleId":
                parents = curr_obj.get("parentReceptacles") or []
                if not isinstance(parents, list):
                    parents = [parents]
                valid_receptacles = SceneSpecialConfig.get_valid_receptacle_ids(scene_name, target_val)
                matched = bool(valid_receptacles.intersection({str(parent) for parent in parents if parent}))
                category_fallback_used = bool(object_result.get("category_fallback_used"))
                if not matched and _normalize_goal_object_type(target_val) == "cabinet":
                    actual_parent_types = {
                        _normalize_goal_object_type(parent)
                        for parent in parents
                        if parent
                    }
                    matched = "cabinet" in actual_parent_types
                    category_fallback_used = matched
                    object_result["category_fallback_used"] = (
                        bool(object_result.get("category_fallback_used")) or category_fallback_used
                    )
                message = ""
                if not matched:
                    message = f"{obj_id} 不在目标容器 {valid_receptacles} 中 (当前在 {parents})"
                    failed_reasons.append(message)
                object_result["checks"].append(
                    {
                        "attribute": attr,
                        "check_type": "parent_receptacle",
                        "expected": target_val,
                        "valid_receptacles": sorted(valid_receptacles),
                        "actual": parents,
                        "passed": matched,
                        "category_fallback_used": category_fallback_used,
                        "message": message,
                    }
                )
            
            # # 3. 检查反向容器 (如抹布不能在浴缸里) - 如果你之前在模板里定义了 forbiddenReceptacle
            # elif attr == "forbiddenReceptacle":
            #     parents = curr_obj.get("parentReceptacles") or[]
            #     if target_val in parents:
            #         failed_reasons.append(f"{obj_id} 错误地留在了被禁止的容器 {target_val} 中")
            
            # 4. 常规物理属性 (isOpen, isDirty, isToggled 等)
            else:
                actual_val, compare_target = _extract_goal_actual_and_expected_value(
                    curr_obj,
                    attr,
                    target_val,
                )
                matched = actual_val == compare_target
                category_fallback_used = bool(object_result.get("category_fallback_used"))
                if not matched and attr == "isOpen" and normalized_goal_type == "cabinet":
                    for candidate in objects_by_type.get("cabinet", []):
                        candidate_actual, candidate_expected = _extract_goal_actual_and_expected_value(
                            candidate,
                            attr,
                            target_val,
                        )
                        if candidate_actual == candidate_expected:
                            actual_val = candidate_actual
                            matched = True
                            category_fallback_used = True
                            object_result["category_fallback_used"] = True
                            break
                message = ""
                if not matched:
                    message = f"{obj_id} 的 {attr} 期望为 {target_val}, 实际为 {actual_val}"
                    failed_reasons.append(message)
                object_result["checks"].append(
                    {
                        "attribute": attr,
                        "check_type": "state",
                        "expected": target_val,
                        "comparison_expected": compare_target,
                        "actual": actual_val,
                        "passed": matched,
                        "category_fallback_used": category_fallback_used,
                        "message": message,
                    }
                )

        object_results.append(object_result)

    return {
        "success": len(failed_reasons) == 0,
        "failed_reasons": failed_reasons,
        "objects": object_results,
        "action_history_actions": action_history_act,
    }


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    return str(value)


def build_request_evaluation_record(
    *,
    task: dict[str, Any],
    final_obs: dict[str, Any],
    goal_detail: dict[str, Any],
    runtime_failure_reasons: list[str] | None = None,
) -> dict[str, Any]:
    request_id = str(task.get("request_id", ""))
    final_obs_without_image = {}
    if isinstance(final_obs, dict):
        final_obs_without_image = {
            key: value
            for key, value in final_obs.items()
            if key != "image"
        }

    target_object_states = {
        item.get("object_id"): item.get("final_object_state")
        for item in (goal_detail.get("objects") or [])
        if item.get("object_id") and item.get("final_object_state") is not None
    }

    goal_success = bool(goal_detail.get("success"))
    runtime_failure_reasons = [
        str(reason).strip()
        for reason in (runtime_failure_reasons or [])
        if str(reason).strip()
    ]
    return {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "request_id": request_id,
        "task_id": task.get("instance_id", ""),
        "scene_name": task.get("scene_name", ""),
        "room_id": task.get("room_id", ""),
        # Benchmark success is defined by final goal state. Runtime/tool failures are
        # preserved below for diagnostics, but they do not override goal-state success.
        "success": goal_success,
        "goal_success": goal_success,
        "runtime_success": not runtime_failure_reasons,
        "runtime_failure_reasons": runtime_failure_reasons,
        "failed_reasons": list(goal_detail.get("failed_reasons") or []),
        "goal_verification": task.get("goal_verification") or {},
        "goal_evaluation": goal_detail,
        "target_object_states": target_object_states,
        "final_observation": final_obs_without_image,
    }


def write_request_evaluation_log(log_dir: Path, record: dict[str, Any]) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    request_id = str(record.get("request_id") or "unknown_request")
    output_path = log_dir / f"{request_id}.json"
    output_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return output_path


def _has_usable_final_request_observation(final_obs: dict[str, Any]) -> bool:
    if not isinstance(final_obs, dict):
        return False
    metadata = final_obs.get("metadata")
    if not isinstance(metadata, dict):
        return False
    objects = metadata.get("objects")
    return isinstance(objects, list) and bool(objects)


def _build_missing_final_observation_goal_detail(
    *,
    task: dict[str, Any],
    final_obs: dict[str, Any],
) -> dict[str, Any]:
    request_id = str(task.get("request_id") or "")
    scene_name = str(task.get("scene_name") or "")
    metadata = final_obs.get("metadata") if isinstance(final_obs, dict) else {}
    warning = metadata.get("agentObservationWarning") if isinstance(metadata, dict) else None
    history = final_obs.get("history") if isinstance(final_obs, dict) else None
    action_history_actions = [
        action.get("action")
        for action in (history or [])
        if isinstance(action, dict) and action.get("action")
    ]
    if isinstance(warning, str) and warning.strip():
        message = f"请求 {request_id} 缺少最终场景观测 (scene={scene_name}): {warning.strip()}"
    else:
        message = f"请求 {request_id} 缺少最终场景观测，可能未被执行或未进入房间 (scene={scene_name})"
    return {
        "success": False,
        "failed_reasons": [message],
        "objects": [],
        "action_history_actions": action_history_actions,
    }


def _finalize_request_bundle_evaluation(
    *,
    env,
    selected_tasks: list[dict[str, Any]],
    run_eval_dir: Path,
    interrupted: bool = False,
    runtime_failure_reason: str | None = None,
    generate_timeline: bool = True,
) -> dict[str, Any]:
    if interrupted:
        print("🔍 检测到运行被中断，开始基于当前环境状态执行最终评测与图片导出 ...")
    else:
        print("🔍 正在进行最终状态评判 ...")

    total_tasks = len(selected_tasks)
    success_tasks = 0
    evaluation_log_dir = run_eval_dir / "request_evaluations"
    evaluation_summary: list[dict[str, Any]] = []

    for task in selected_tasks:
        scene_name = task["scene_name"]
        task_id = task.get("instance_id", "")
        request_id = task["request_id"]
        goal_spec = task.get("goal_verification")
        final_obs = env.get_request_observation(request_id)
        metadata = final_obs.get("metadata") if isinstance(final_obs, dict) else {}
        metadata_objects = metadata.get("objects") if isinstance(metadata, dict) else []
        action_history = final_obs.get("history") if isinstance(final_obs, dict) else []

        if isinstance(final_obs.get("image"), (list, tuple)):
            pass
        elif final_obs.get("image") is not None:
            req_img_dir = run_eval_dir / "env_images"
            req_img_dir.mkdir(parents=True, exist_ok=True)
            init_img_path = req_img_dir / f"{request_id}_final.png"
            Image.fromarray(final_obs["image"]).save(init_img_path)

        if _has_usable_final_request_observation(final_obs):
            goal_detail = evaluate_goal_detailed(
                goal_verification=goal_spec,
                metadata_objects=metadata_objects,
                action_history=action_history,
                scene_name=scene_name,
            )
        else:
            goal_detail = _build_missing_final_observation_goal_detail(
                task=task,
                final_obs=final_obs,
            )

        runtime_failure_reasons = []
        if runtime_failure_reason:
            runtime_failure_reasons = [runtime_failure_reason]

        is_success = bool(goal_detail["success"])
        reasons = list(goal_detail["failed_reasons"])
        evaluation_record = build_request_evaluation_record(
            task=task,
            final_obs=final_obs,
            goal_detail=goal_detail,
            runtime_failure_reasons=runtime_failure_reasons,
        )
        evaluation_log_path = write_request_evaluation_log(evaluation_log_dir, evaluation_record)
        evaluation_summary.append(
            {
                "request_id": request_id,
                "task_id": task_id,
                "scene_name": scene_name,
                "success": is_success,
                "goal_success": is_success,
                "runtime_success": not runtime_failure_reasons,
                "runtime_failure_reasons": runtime_failure_reasons,
                "failed_reasons": reasons,
                "evaluation_log": str(evaluation_log_path.name),
            }
        )

        if is_success:
            print(f"🏆 结果: 成功 (Success)! request_id={request_id} task={task_id} scene={scene_name}")
            success_tasks += 1
        else:
            print(f"💥 结果: 失败 (Failed). request_id={request_id} task={task_id} scene={scene_name} 原因:")
            for r in reasons:
                print(f"    - {r}")
        if runtime_failure_reasons:
            print(f"⚠️ 运行诊断: request_id={request_id} runtime failure 不计入 goal-state 成功率:")
            for r in runtime_failure_reasons:
                print(f"    - {r}")
        print(f"📝 评判详情日志: {evaluation_log_path}")

    summary_path = evaluation_log_dir / "_summary.json"
    summary_path.write_text(
        json.dumps(evaluation_summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )

    sr = (success_tasks / total_tasks * 100.0) if total_tasks else 0.0
    print(f"\n📊 评测结束! 总任务数: {total_tasks}, 成功数: {success_tasks}, 成功率 (SR): {sr:.2f}%")
    if generate_timeline:
        _generate_eval_timeline_plots(run_eval_dir)
    return {
        "interrupted": bool(interrupted),
        "total_tasks": total_tasks,
        "success_tasks": success_tasks,
        "success_rate": sr,
        "evaluation_summary": evaluation_summary,
    }


def _build_sequential_request_message(
    task: dict[str, Any],
    *,
    order: int,
    total: int,
    request_type: str | None = None,
) -> str:
    intro = f"Sequential request {order} of {total}:"
    return _build_request_message([task], intro=intro, request_type=request_type)


def _agent_request_completed(agent, request_id: str) -> bool:
    memory_store = getattr(agent, "memory", None)
    if memory_store is None or not request_id:
        return False
    try:
        active_requests = memory_store.get_memory().get("active_requests") or []
    except Exception:
        return False

    for entry in active_requests:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("request_id") or "") != request_id:
            continue
        if str(entry.get("overall_status") or "pending") == "completed":
            return True

        request_progress = entry.get("request_progress")
        if isinstance(request_progress, dict) and request_progress:
            terminal_statuses = {"completed", "skipped_already_satisfied"}
            return all(
                isinstance(item, dict)
                and str(item.get("status") or "").strip() in terminal_statuses
                for item in request_progress.values()
            )
        return False
    return False


def _agent_has_running_robot_work(agent) -> bool:
    tools = getattr(agent, "tools", None)
    if tools is not None:
        try:
            if tools.has_running_robot_plans():
                return True
        except Exception:
            pass

    memory_store = getattr(agent, "memory", None)
    if memory_store is None:
        return False
    try:
        robots_status = memory_store.get_memory().get("robots_status") or []
    except Exception:
        return False

    for robot in robots_status:
        if not isinstance(robot, dict):
            continue
        current_action = robot.get("current_action")
        current_action_text = str(current_action or "").strip().lower()
        if current_action_text not in {"", "none", "null"}:
            return True
    return False


def _sequential_gate_failure_reason(agent, request_id: str) -> str | None:
    if not _agent_request_completed(agent, request_id):
        return f"{request_id}: previous sequential request is not completed"
    if _agent_has_running_robot_work(agent):
        return f"{request_id}: robot work is still running after request completion"
    return None


def _mark_baseline_viki_request_finished(agent, request_id: str) -> None:
    """Baseline-Viki has no memory tools, so the eval driver closes its request gate.

    The final goal evaluator remains authoritative for success/failure. This only
    prevents sequential delivery from blocking forever on agent-managed memory
    fields that Baseline-Viki intentionally cannot update.
    """
    memory_store = getattr(agent, "memory", None)
    if memory_store is None or not request_id:
        return
    try:
        memory_store.update_request_status(
            request_id=request_id,
            overall_status="completed",
        )
    except Exception:
        return


def _execute_baseline_viki_sequential_requests(
    *,
    agent,
    selected_tasks: list[dict[str, Any]],
    request_type: str,
    active_skill: str,
) -> str | None:
    runtime_failures: list[str] = []
    total = len(selected_tasks)

    for order, task in enumerate(selected_tasks, start=1):
        if order > 1:
            agent.register_request_tasks(
                [task],
                request_type=request_type,
                active_skill=active_skill,
                replace=False,
                preserve_robot_assignments=True,
            )

        request_message = _build_sequential_request_message(
            task,
            order=order,
            total=total,
            request_type=request_type,
        )
        agent.process_request_messages(
            request=request_message,
            active_skill=active_skill,
            request_injections=None,
        )

        runtime_failure_reason = getattr(agent, "runtime_failure_reason", None)
        request_id = _get_request_id_from_task(task, order)
        if runtime_failure_reason:
            runtime_failures.append(f"{request_id}: {runtime_failure_reason}")
            _mark_baseline_viki_request_finished(agent, request_id)
            continue

        if _agent_has_running_robot_work(agent):
            runtime_failures.append(f"{request_id}: robot work is still running after request execution")
            _mark_baseline_viki_request_finished(agent, request_id)
            continue
        _mark_baseline_viki_request_finished(agent, request_id)

    if runtime_failures:
        return "Baseline-Viki sequential request execution failed: " + " | ".join(runtime_failures)
    return None


def _execute_agent_sequential_requests(
    *,
    agent,
    selected_tasks: list[dict[str, Any]],
    request_type: str,
    active_skill: str,
) -> str | None:
    runtime_failures: list[str] = []
    total = len(selected_tasks)

    for order, task in enumerate(selected_tasks, start=1):
        if order > 1:
            agent.register_request_tasks(
                [task],
                request_type=request_type,
                active_skill=active_skill,
                replace=False,
                preserve_robot_assignments=True,
            )

        request_message = _build_sequential_request_message(
            task,
            order=order,
            total=total,
            request_type=request_type,
        )
        agent.process_request_messages(
            request=request_message,
            active_skill=active_skill,
            request_injections=None,
        )

        runtime_failure_reason = getattr(agent, "runtime_failure_reason", None)
        if runtime_failure_reason:
            request_id = str(task.get("request_id") or f"request_{order}")
            runtime_failures.append(f"{request_id}: {runtime_failure_reason}")
            break

        request_id = _get_request_id_from_task(task, order)
        gate_failure = _sequential_gate_failure_reason(agent, request_id)
        if gate_failure:
            runtime_failures.append(gate_failure)
            break

    if runtime_failures:
        return "Sequential request execution failed: " + " | ".join(runtime_failures)
    return None


def _execute_request_bundle_with_finalization(
    *,
    agent,
    env,
    request: str,
    active_skill: str,
    request_injections: list[dict[str, Any]],
    selected_tasks: list[dict[str, Any]],
    run_eval_dir: Path,
    agent_mode: str = AGENT_MODE_EMBODIEDCOOP,
    request_type: str = DEFAULT_REQUEST_TYPE,
    request_delivery_mode: str = "simultaneous",
) -> dict[str, Any]:
    interrupted = False

    try:
        if agent_mode == AGENT_MODE_BASELINE_VIKI:
            sequential_runtime_failure = _execute_baseline_viki_sequential_requests(
                agent=agent,
                selected_tasks=selected_tasks,
                request_type=request_type,
                active_skill=active_skill,
            )
            if sequential_runtime_failure:
                agent.runtime_failure_reason = sequential_runtime_failure
        elif request_delivery_mode == "sequential":
            sequential_runtime_failure = _execute_agent_sequential_requests(
                agent=agent,
                selected_tasks=selected_tasks,
                request_type=request_type,
                active_skill=active_skill,
            )
            if sequential_runtime_failure:
                agent.runtime_failure_reason = sequential_runtime_failure
        else:
            agent.process_request_messages(
                request=request,
                active_skill=active_skill,
                request_injections=list(request_injections),
            )
    except KeyboardInterrupt:
        interrupted = True
        print("[EmbodiedCoop][INTERRUPTED] 收到 KeyboardInterrupt，继续执行最终 goal 评测与图片导出。")

    runtime_failure_reason = getattr(agent, "runtime_failure_reason", None)
    return _finalize_request_bundle_evaluation(
        env=env,
        selected_tasks=selected_tasks,
        run_eval_dir=run_eval_dir,
        interrupted=interrupted,
        runtime_failure_reason=runtime_failure_reason,
        generate_timeline=True,
    )




def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    把 YAML/JSON config 转成参数，并允许 CLI 覆盖
    """
    parser = argparse.ArgumentParser(description="Evaluate EmbodiedCoop with request/robot-team setup")
    parser.add_argument(
        "--scene",
        type=str,
        default="",
        help="Optional scene key in --eval-bundles-file, e.g. hotel_housekeeping, facility_inspection, restaurant_serving, or restocking.",
    )
    parser.add_argument(
        "--eval-bundles-file",
        type=str,
        default=DEFAULT_EVAL_BUNDLES_FILE,
        help="JSON file that maps eval scenes and bundle ids to task/bundle config.",
    )
    parser.add_argument(
        "--agent-mode",
        type=str,
        choices=[
            AGENT_MODE_EMBODIEDCOOP,
            AGENT_MODE_BASELINE_VIKI,
            AGENT_MODE_BASELINE_ROBOOS,
        ],
        default=os.getenv("EMBODIEDCOOP_AGENT_MODE", AGENT_MODE_EMBODIEDCOOP),
        help=(
            "Agent runtime mode. baseline-viki receives bundle requests sequentially. "
            "baseline-roboos decomposes the currently visible request set into a DAG, executes by wave, and may replan the remaining DAG as streamed requests arrive."
        ),
    )
    parser.add_argument(
        "--task-json",
        type=str,
        default=DEFAULT_TASK_JSON,
        help="Task JSON path",
    )

    # request bundle 文件目录，程序会按 req 数和 request-bundle-file-prefix 拼文件名读取。
    parser.add_argument(
        "--request-bundle-dir",
        type=str,
        default=DEFAULT_REQUEST_BUNDLE_DIR,
        help="Directory that contains {request_bundle_file_prefix}_req{count}.json files.",
    )

    parser.add_argument(
        "--robot-team-preset",
        type=str,
        default="heterogeneous_robot1_no_clean_2",
        help="Robot team preset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("EMBODIEDCOOP_MODEL", "gemini-3-flash-preview-high"),
        help="LLM model name used for the eval run, e.g. gemini-3-flash-preview-high or qwen3.6-27b.",
    )
    parser.add_argument(
        "--log-skill-name",
        type=str,
        default=os.getenv("EMBODIEDCOOP_LOG_SKILL_NAME", DEFAULT_LOG_SKILL_NAME),
        help=f"Folder label used to group eval logs by skill, e.g. {DEFAULT_LOG_SKILL_NAME}.",
    )
    parser.add_argument(
        "--request-type",
        type=str,
        default=os.getenv("EMBODIEDCOOP_REQUEST_TYPE", DEFAULT_REQUEST_TYPE),
        help="Logical request type used to initialize request memory.",
    )
    parser.add_argument(
        "--active-skill",
        type=str,
        default=os.getenv("EMBODIEDCOOP_ACTIVE_SKILL", DEFAULT_ACTIVE_SKILL),
        help="Skill directory/name loaded into the agent context for this eval run.",
    )
    parser.add_argument(
        "--request-bundle-count",
        type=int,
        default=2,
        help="Eval mode only: pick req{count} file, e.g. req2/req3/req4/req5.",
    )
    parser.add_argument(
        "--request-bundle-index",
        type=int,
        default=0,
        help="Eval mode only: pick bundle index inside the req{count} file.",
    )
    parser.add_argument(
        "--request-bundle-id",
        type=str,
        default="",
        help="Eval mode only: pick bundle by bundle_id (e.g. req3_sum19_bundle_000003). Takes priority over --request-bundle-index.",
    )
    parser.add_argument(
        "--bundle-id",
        type=str,
        default="",
        help="Alias for --request-bundle-id when using a scene-based eval config.",
    )
    parser.add_argument(
        "--request-bundle-file-prefix",
        type=str,
        default=os.getenv("EMBODIEDCOOP_REQUEST_BUNDLE_FILE_PREFIX", DEFAULT_REQUEST_BUNDLE_FILE_PREFIX),
        help="Prefix for request bundle files under --request-bundle-dir, e.g. tasks_facility_inspection_bundles.",
    )
    parser.add_argument(
        "--max-iterations",
        type=_parse_positive_int_or_auto,
        default=_parse_positive_int_or_auto("auto"),
        help="Max agent rounds. Use a positive integer to force a fixed cap, or 'auto' to derive it from the selected bundle's total pddl plan length. When full-memory ablation is enabled and matching successful baseline runs exist, auto is additionally capped by baseline completion rounds + a small buffer.",
    )
    parser.add_argument(
        "--request-delivery-mode",
        type=str,
        choices=["simultaneous", "incremental", "sequential"],
        default=os.getenv("EMBODIEDCOOP_REQUEST_DELIVERY_MODE", "simultaneous"),
        help=(
            "How to deliver bundle requests to the agent: all at once, injected incrementally during execution, "
            "or sequentially after the previous request-processing call returns."
        ),
    )
    parser.add_argument(
        "--initial-request-count",
        type=int,
        default=int(os.getenv("EMBODIEDCOOP_INITIAL_REQUEST_COUNT", "1")),
        help="When request-delivery-mode=incremental, how many requests are included in the first user message.",
    )
    parser.add_argument(
        "--request-injection-interval-rounds",
        type=_parse_positive_int_or_range,
        default=_parse_positive_int_or_range(os.getenv("EMBODIEDCOOP_REQUEST_INJECTION_INTERVAL_ROUNDS", "1")),
        help="When request-delivery-mode=incremental, inject queued request batches after sampled intervals. Supports a fixed integer like 3 or a random range like 1-4; range values produce one sampled interval per injection batch and the generated schedule is reused for the same bundle.",
    )
    parser.add_argument(
        "--request-injection-batch-size",
        type=int,
        default=int(os.getenv("EMBODIEDCOOP_REQUEST_INJECTION_BATCH_SIZE", "1")),
        help="When request-delivery-mode=incremental, how many queued requests to inject in each additional user message.",
    )
    parser.add_argument(
        "--ablate-all-memory",
        action="store_true",
        help="Ablation: remove all model-visible memory context and disable all memory tools while keeping internal logs/artifacts.",
    )
    parser.add_argument(
        "--ablate-agent-memory",
        action="store_true",
        help=(
            "Ablation: keep framework-managed robots_status and scene_state visible, "
            "but hide agent-managed progress/status/history memory and disable memory update tools."
        ),
    )


    # 打印当前可用 robot preset 名称，方便你手工选
    parser.add_argument(
        "--list-robot-team-presets",
        action="store_true",
        help="List available robot team presets.",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=(
            "Path to a YAML or JSON config file that provides default eval parameters. "
            "Command-line arguments override values from the config file."
        ),
    )
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config-file", type=str, default=None)
    config_args, _ = config_parser.parse_known_args(raw_argv)
    config_dict = _load_config_file(config_args.config_file)
    scene_cli_overrides = _extract_scene_cli_overrides(raw_argv)
    if scene_cli_overrides:
        config_dict = {**config_dict, **scene_cli_overrides}
    config_dict = _expand_scene_config(config_dict)
    config_argv = _config_dict_to_argv(config_dict)
    effective_argv = config_argv + raw_argv
    args = parser.parse_args(effective_argv)
    if args.bundle_id and not args.request_bundle_id:
        args.request_bundle_id = args.bundle_id
    return args


def main():
    args = parse_args()

    task_json_path = args.task_json

    # 请求采样 + robot team 的统一设置
    setup = TaskAndTeamSetup(
        task_json_path=task_json_path,
        get_robot_team_config_fn=get_robot_team_config,
        list_robot_team_presets_fn=list_robot_team_presets,
    )

    available_presets = setup.list_robot_team_presets()
    if args.list_robot_team_presets:
        print(f"[EmbodiedCoop][RobotTeam] available_presets={available_presets}")
        return

    # 1. 加载任务数据
    with open(task_json_path, "r") as f:
        all_tasks = json.load(f)


    # 2. robot team 配置 和 request 配置
    preset_name = args.robot_team_preset
    team_config = setup.get_robot_team(preset_name)
    robot_ids = team_config["robot_ids"]
    robot_action_sets = team_config.get("robot_action_sets")

    request_bundle_file = Path(args.request_bundle_dir) / (
        f"{args.request_bundle_file_prefix}_req{args.request_bundle_count}.json"
    )
    request_bundle = setup.load_request_bundle(
        bundles_path=request_bundle_file,
        bundle_index=args.request_bundle_index,
        bundle_id=args.request_bundle_id or None,
    )

    _bundle_label = (args.request_bundle_id.strip() or None) if args.request_bundle_id else None
    _bundle_label = _bundle_label or f"idx{args.request_bundle_index}"

    print(
        "[EmbodiedCoop][RobotTeam] "
        f"preset={preset_name} description={team_config.get('description', '')} "
        f"robot_ids={robot_ids} "
        f"mode={_robot_team_mode(robot_action_sets)}"
    )
    # print(f"[EmbodiedCoop][RobotTeam] available_presets={available_presets}")
    print(
        "[EmbodiedCoop][RequestBundle] "
        f"source={request_bundle_file} bundle={_bundle_label} "
        f"request_count={request_bundle['request_count']} "
        f"request_ids={request_bundle['request_ids']}"
    )

    # 3. 保存信息设置
    # 默认使用当前工作目录；可通过环境变量 EMBODIEDCOOP_WORKSPACE 覆盖为其他目录
    workspace_dir = Path(os.getenv("EMBODIEDCOOP_WORKSPACE", ".")).resolve()

    # 每次 eval/task 单独输出目录（消息、图片、memory）
    request_type = args.request_type
    active_skill = args.active_skill
    model = args.model
    effective_request_delivery_mode, request_delivery_override_reason = _resolve_effective_request_delivery_mode(
        agent_mode=args.agent_mode,
        requested_mode=args.request_delivery_mode,
    )
    experiment_category, experiment_name = _build_experiment_log_group(
        agent_mode=args.agent_mode,
        ablate_all_memory=args.ablate_all_memory,
        ablate_agent_memory=args.ablate_agent_memory,
    )
    eval_group_dir = _build_eval_group_dir(
        workspace_dir,
        model=model,
        experiment_category=experiment_category,
        experiment_name=experiment_name,
        skill_name=args.log_skill_name,
    )
    eval_group_dir.mkdir(parents=True, exist_ok=True)

    # 4. 从 request bundle 中解析全部 task（single/multi 统一入口）
    selected_tasks: list[dict] = []
    for request_meta in request_bundle.get("requests", []):
        scene_name = request_meta.get("scene_name")
        task_id = request_meta.get("instance_id")
        if not scene_name or not task_id:
            print(f"[EmbodiedCoop][WARN] invalid request_meta={request_meta}, skip")
            continue

        task = None
        scene_tasks = all_tasks.get(scene_name, [])
        for t in scene_tasks:
            if isinstance(t, dict) and t.get("instance_id") == task_id:
                task = setup.normalize_task_instance(dict(t))
                break
        if task is None:
            print(f"[EmbodiedCoop][WARN] task not found for instance_id={task_id}, skip")
            continue

        task.setdefault("request_id", request_meta.get("request_id", f"req_{task_id}"))
        room_id = request_meta.get("room_id") or str(scene_name).replace("FloorPlan", "")
        task.setdefault("room_id", str(room_id))
        task.setdefault("scene_name", str(scene_name))
        selected_tasks.append(task)

    if not selected_tasks:
        print("[EmbodiedCoop][ERROR] no valid tasks selected from request bundle")
        return
    
    # 5. 初始化多房间环境（single-room 时同样兼容）
    env = MultiRoomThorEnv(env_factory=ThorSymbolicEnv)
    env.register_task_contexts(selected_tasks)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_eval_dir = eval_group_dir / f"bundle_req{request_bundle.get('request_count', len(selected_tasks))}_{_bundle_label}_{run_stamp}"
    run_eval_dir.mkdir(parents=True, exist_ok=True)

    bundle_dump = run_eval_dir / "selected_request_bundle.json"
    bundle_dump.write_text(json.dumps(request_bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[EmbodiedCoop][RequestBundle] saved={bundle_dump}")

    memory_file = run_eval_dir / "memory.json"
    print(f"[EmbodiedCoop][Memory] initialized_memory_file={memory_file.resolve()}")

    max_iterations, max_iterations_label = _resolve_agent_max_iterations(
        args.max_iterations,
        request_bundle=request_bundle,
        selected_tasks=selected_tasks,
        robot_ids=robot_ids,
        workspace_dir=workspace_dir,
        model=model,
        skill_name=args.log_skill_name,
        use_baseline_reference_cap=bool(args.ablate_all_memory or args.ablate_agent_memory),
    )
    print(f"[EmbodiedCoop][AgentBudget] max_iterations={max_iterations} source={max_iterations_label}")

    interval_rounds = "1"
    interval_rounds_label = "n/a"
    cached_delivery_plan = None
    if effective_request_delivery_mode == "incremental":
        interval_rounds, interval_rounds_label = _resolve_bundle_request_injection_interval_rounds(
            args.request_injection_interval_rounds,
            workspace_dir=workspace_dir,
            request_bundle_file=request_bundle_file,
            request_bundle=request_bundle,
            bundle_label=_bundle_label,
        )
        _, _, _, interval_cache_entry = _load_bundle_request_interval_cache_context(
            workspace_dir=workspace_dir,
            request_bundle_file=request_bundle_file,
            request_bundle=request_bundle,
            bundle_label=_bundle_label,
        )
        cached_delivery_plan = interval_cache_entry.get("request_delivery_plan") if isinstance(interval_cache_entry, dict) else None

     
    # 6. 初始化 agent
    agent_class = _resolve_agent_class(args.agent_mode)
    agent_kwargs = {
        "model": model,
        "workspace": workspace_dir,
        "max_iterations": max_iterations,
        "robot_ids": robot_ids,
        "robot_action_sets": robot_action_sets,
        "memory_file": memory_file,
        "eval_artifact_dir": run_eval_dir / "agent_rounds",
        "ablate_all_memory": args.ablate_all_memory,
        "ablate_agent_memory": args.ablate_agent_memory,
    }
    extra_actions = _resolve_extra_actions_for_robot_team(
        active_skill=active_skill,
        robot_action_sets=robot_action_sets,
    )
    if extra_actions:
        agent_kwargs["extra_actions"] = extra_actions
    agent = agent_class(**agent_kwargs)

    # reset 用于清理上轮状态；随后覆盖为多任务上下文
    if effective_request_delivery_mode == "sequential":
        initial_tasks = [selected_tasks[0]]
        initial_request_message = _build_sequential_request_message(
            selected_tasks[0],
            order=1,
            total=len(selected_tasks),
            request_type=request_type,
        )
        request_injections = []
        request_delivery_plan_source = "sequential_after_completion"
    elif effective_request_delivery_mode == "simultaneous":
        initial_tasks = list(selected_tasks)
        initial_request_message = request_bundle.get("combined_request_text") or _build_request_message(
            initial_tasks,
            request_type=request_type,
        )
        request_injections = []
        request_delivery_plan_source = "direct_delivery"
    else:
        initial_tasks, initial_request_message, request_injections, request_delivery_plan_source = _build_request_delivery_plan(
            selected_tasks=selected_tasks,
            request_bundle=request_bundle,
            delivery_mode=effective_request_delivery_mode,
            initial_request_count=args.initial_request_count,
            injection_interval_rounds=interval_rounds,
            injection_batch_size=args.request_injection_batch_size,
            request_type=request_type,
            active_skill=active_skill,
            cached_delivery_plan=cached_delivery_plan,
        )
        if request_delivery_plan_source == "generated_bundle_schedule":
            _persist_bundle_request_delivery_plan(
                workspace_dir=workspace_dir,
                request_bundle_file=request_bundle_file,
                request_bundle=request_bundle,
                bundle_label=_bundle_label,
                selected_tasks=selected_tasks,
                initial_tasks=initial_tasks,
                request_injections=request_injections,
                interval_spec=interval_rounds,
                initial_request_count=args.initial_request_count,
                injection_batch_size=args.request_injection_batch_size,
                source_spec=args.request_injection_interval_rounds,
            )
    interval_sequence = _derive_request_injection_interval_sequence(request_injections)
    if interval_sequence:
        interval_rounds_label = f"{interval_rounds_label}; sampled_sequence={interval_sequence}; plan={request_delivery_plan_source}"
    else:
        interval_rounds_label = f"{interval_rounds_label}; plan={request_delivery_plan_source}"
    print(f"[EmbodiedCoop][RequestDelivery] interval_spec={interval_rounds} source={interval_rounds_label}")

    run_metadata = {
        "run_stamp": run_stamp,
        "run_eval_dir": str(run_eval_dir),
        "agent_mode": args.agent_mode,
        "model": model,
        "log_skill_name": args.log_skill_name,
        "request_type": request_type,
        "active_skill": active_skill,
        "task_json_path": str(task_json_path),
        "request_bundle_file": str(request_bundle_file),
        "request_bundle_id": _bundle_label,
        "request_bundle_count": request_bundle.get("request_count", len(selected_tasks)),
        "request_ids": request_bundle.get("request_ids"),
        "selected_task_count": len(selected_tasks),
        "selected_tasks": [
            {
                "request_id": task.get("request_id"),
                "instance_id": task.get("instance_id"),
                "scene_name": task.get("scene_name"),
                "room_id": task.get("room_id"),
            }
            for task in selected_tasks
        ],
        "robot_team_preset": preset_name,
        "robot_team_description": team_config.get("description", ""),
        "robot_team_mode": _robot_team_mode(robot_action_sets),
        "robot_ids": robot_ids,
        "robot_action_sets": robot_action_sets,
        "max_iterations": max_iterations,
        "max_iterations_source": max_iterations_label,
        "request_delivery_mode": effective_request_delivery_mode,
        "requested_delivery_mode": args.request_delivery_mode,
        "request_delivery_override_reason": request_delivery_override_reason,
        "request_delivery_plan_source": request_delivery_plan_source,
        "initial_request_count": len(initial_tasks),
        "initial_request_ids": [task.get("request_id") for task in initial_tasks],
        "request_injection_count": len(request_injections),
        "request_injection_interval_rounds": interval_rounds,
        "request_injection_interval_rounds_source": interval_rounds_label,
        "request_injection_batch_size": args.request_injection_batch_size,
        "ablate_all_memory": args.ablate_all_memory,
        "ablate_agent_memory": args.ablate_agent_memory,
    }
    metadata_path = run_eval_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(f"[EmbodiedCoop][RunMetadata] saved={metadata_path}")

    agent.reset(initial_tasks[0], env=env)
    agent.register_request_tasks(
        initial_tasks,
        request_type=request_type,
        active_skill=active_skill,
        replace=True,
        preserve_robot_assignments=False,
    )

    _print_eval_run_overview(
        args=args,
        agent_mode=args.agent_mode,
        task_json_path=task_json_path,
        preset_name=preset_name,
        team_config=team_config,
        robot_ids=robot_ids,
        robot_action_sets=robot_action_sets,
        request_bundle_file=request_bundle_file,
        request_bundle=request_bundle,
        bundle_label=_bundle_label,
        selected_tasks=selected_tasks,
        model=model,
        request_type=request_type,
        active_skill=active_skill,
        max_iterations=max_iterations,
        max_iterations_label=max_iterations_label,
        run_eval_dir=run_eval_dir,
        memory_file=memory_file,
        interval_rounds_label=interval_rounds_label,
        request_injections=request_injections,
        initial_tasks=initial_tasks,
        effective_request_delivery_mode=effective_request_delivery_mode,
        request_delivery_override_reason=request_delivery_override_reason,
    )

    # 7. Agent 执行（single/multi 均发送组合请求）
    print(f"\n=======================================================")
    print(
        "🎬 开始执行任务组合: "
        f"bundle_request_count={request_bundle.get('request_count', len(selected_tasks))} "
        f"selected_tasks={len(selected_tasks)}"
    )
    print(
        "[EmbodiedCoop][Delivery] "
        f"mode={effective_request_delivery_mode} initial_requests={len(initial_tasks)} "
        f"pending_injections={len(request_injections)} interval_schedule={interval_rounds_label} "
        f"batch_size={args.request_injection_batch_size}"
    )
    _execute_request_bundle_with_finalization(
        agent=agent,
        env=env,
        request=initial_request_message,
        active_skill=active_skill,
        request_type=request_type,
        request_injections=request_injections,
        selected_tasks=selected_tasks,
        run_eval_dir=run_eval_dir,
        agent_mode=args.agent_mode,
        request_delivery_mode=effective_request_delivery_mode,
    )

if __name__ == "__main__":
    main()
