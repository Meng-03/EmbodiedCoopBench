from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from agents.embodiedclaw.contextbuilder import ContextBuilder
from agents.embodiedclaw.embodiedclaw import EmbodiedClawAgent
from agents.embodiedclaw.agent_selfknowledge.whoiam import build_viki_agent_self_knowledge


_BASELINE_PROMPT_APPENDIX = """## Notice
- You are running a single-shot, full-information baseline.
- Requests are delivered sequentially. Plan and execute only the currently visible request; later requests will be delivered after this request's plan has run.
- Baseline-Viki memory may retain earlier requests for audit/history. Any request whose `baseline_viki_scope` is `history_locked`, whose `baseline_viki_locked` is true, or whose `overall_status` is `completed` is historical context only. Do not plan, repair, continue, or emit tool calls for those requests even if their `scene_state` still shows unmet checklist items.
- The only request you should act on is the one marked `baseline_viki_scope: current` and named by the latest user message.
- Your first assistant response for each request must contain the complete executable tool-call plan for that current request.
- You may emit multiple robot tool calls in a single response, but the runtime executes them strictly one-by-one in the order you emit them.
- Do not rely on parallel execution or simultaneous robot actions. If two actions must both happen, list them in the exact global order they should run.
- You MUST use the API tool-call/function-call channel for every robot action. A plain-text list such as `robot_0__navigate_to(...)` does not execute and counts as a failed plan.
- Do not stop after describing the plan. The response must include at least one robot action tool call whenever any checklist item still requires action.
- Do not reserve work for a later replanning turn. After execution starts, the runtime will not ask you to revise the plan.
- When a later action depends on an earlier deterministic action in your own emitted sequence, include the whole ordered sequence now.
- In request memory, the current request's `scene_state` has already been synchronized from complete room metadata for the objects relevant to that request.
- Treat `scene_state` as the authoritative full-observation snapshot for parent, containment, and object state.
"""

_BASELINE_REFLECT_PROMPT_APPENDIX = """## Reflective Baseline Extension
- If the runtime reports tool failures, treat the failure text as hard feedback and replan from the current state.
- Use the latest successful tool results and the current memory state to revise the remaining plan.
- Avoid repeating actions that already succeeded unless they are required to recover from the failure.
- You will get up to three reflection chances after the initial plan.
"""

class BaselineVikiContextBuilder(ContextBuilder):
    def _get_selfknowledge(self):
        return build_viki_agent_self_knowledge()

    def _include_memory_tools(self) -> bool:
        return False

    def _load_memory_update_guidelines(self) -> str:
        return ""

    def build_system_prompt(self, active_skill: str | None = None):
        prompt = super().build_system_prompt(active_skill=active_skill)
        prompt = prompt.replace(
            "- Constraint: Assign a MAXIMUM of ONE action tool per available robot in a single turn.",
            "- Baseline override: You may assign multiple ordered action tools per robot in a single turn.",
        )
        prompt = prompt.replace(
            "- Sequential Dependency Rule: If a planned action depends on the outcome of a previous action, wait for the feedback before issuing the next action batch.",
            "- Baseline override: Do not wait for feedback. Emit the full ordered plan now when dependencies are already determined by the known request context and your own emitted sequence.",
        )
        prompt = prompt.replace(
            "- You may issue multiple tool calls in one turn when actions are independent and executable.",
            "- You may issue multiple tool calls in one turn, including multi-step FIFO sequences for the same robot when the sequence is executable from known context.",
        )
        return f"{prompt}\n\n---\n\n{_BASELINE_PROMPT_APPENDIX}"


class BaselineVikiReflectContextBuilder(BaselineVikiContextBuilder):
    def build_system_prompt(self, active_skill: str | None = None):
        prompt = super().build_system_prompt(active_skill=active_skill)
        return f"{prompt}\n\n---\n\n{_BASELINE_REFLECT_PROMPT_APPENDIX}"


class BaselineVikiAgent(EmbodiedClawAgent):
    """One-shot full-information baseline built on top of EmbodiedClaw runtime."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = BaselineVikiContextBuilder(self.workspace)
        self.context.memory = self.memory
        self.context.set_tool_registry(self.tools)
        self.runtime_failure_reason: str | None = None
        self._baseline_round_offset = 0

    def reset(self, task_instance, env=None):
        super().reset(task_instance, env=env)
        self.runtime_failure_reason = None
        self._baseline_round_offset = 0

    def register_request_tasks(
        self,
        task_instances: list[dict[str, Any]] | None,
        *,
        request_type: str | None = None,
        active_skill: str | None = None,
        replace: bool = False,
        preserve_robot_assignments: bool = True,
    ) -> list[dict[str, Any]]:
        normalized_tasks = [dict(task) for task in (task_instances or []) if isinstance(task, dict)]
        registered = super().register_request_tasks(
            normalized_tasks,
            request_type=request_type,
            active_skill=active_skill,
            replace=replace,
            preserve_robot_assignments=preserve_robot_assignments,
        )
        self._seed_full_known_request_memory(normalized_tasks)
        self._mark_baseline_viki_request_scope(normalized_tasks)
        return registered

    def _mark_baseline_viki_request_scope(self, current_tasks: list[dict[str, Any]]) -> None:
        """Keep prior sequential requests as locked history in Baseline-Viki prompts."""
        current_request_ids = {
            request_id
            for task in current_tasks
            if (request_id := self._task_request_id(task))
        }
        if not current_request_ids:
            return

        memory = self.memory.get_memory()
        active_requests = memory.get("active_requests")
        if not isinstance(active_requests, list):
            return

        changed = False
        for entry in active_requests:
            if not isinstance(entry, dict):
                continue
            request_id = str(entry.get("request_id") or "")
            if request_id in current_request_ids:
                updates = {
                    "baseline_viki_scope": "current",
                    "baseline_viki_locked": False,
                    "baseline_viki_instruction": "Plan and emit tool calls only for this current sequential request.",
                }
            else:
                updates = {
                    "overall_status": "completed",
                    "baseline_viki_scope": "history_locked",
                    "baseline_viki_locked": True,
                    "baseline_viki_instruction": (
                        "Historical Baseline-Viki request retained for audit only; "
                        "ignore it for planning and do not emit tool calls for it."
                    ),
                }

            for key, value in updates.items():
                if entry.get(key) != value:
                    entry[key] = value
                    changed = True

        if changed:
            self.memory._save_memory(memory)  # pylint: disable=protected-access

    def _seed_full_known_request_memory(self, task_instances: list[dict[str, Any]]) -> None:
        if not task_instances:
            return

        for task in task_instances:
            request_id = self._task_request_id(task)
            if not request_id:
                continue

            observation = self._get_full_known_request_observation(
                request_id=request_id,
                room_id=task.get("room_id"),
            )
            metadata = observation.get("metadata") if isinstance(observation, dict) else None
            metadata_objects = metadata.get("objects") if isinstance(metadata, dict) else None
            if not isinstance(metadata_objects, list) or not metadata_objects:
                continue

            self._sync_request_scene_state_from_full_observation(
                request_id=request_id,
                observation=observation,
                fallback_scene_name=str(task.get("scene_name") or task.get("room_id") or ""),
            )

    def _get_full_known_request_observation(
        self,
        *,
        request_id: str,
        room_id: str | None = None,
    ) -> dict[str, Any]:
        env = getattr(self.tools, "env", None)
        if env is None:
            return {}

        if hasattr(env, "get_request_observation"):
            try:
                observation = env.get_request_observation(request_id)
            except Exception:
                observation = None
            if isinstance(observation, dict):
                return observation

        if room_id is not None and hasattr(env, "get_room_observation"):
            try:
                observation = env.get_room_observation(room_id=str(room_id), request_id=request_id)
            except Exception:
                observation = None
            if isinstance(observation, dict):
                return observation

        return {}

    def _sync_request_scene_state_from_full_observation(
        self,
        *,
        request_id: str,
        observation: dict[str, Any],
        fallback_scene_name: str = "",
    ) -> None:
        metadata = observation.get("metadata") if isinstance(observation, dict) else None
        metadata_objects = metadata.get("objects") if isinstance(metadata, dict) else None
        if not isinstance(metadata_objects, list) or not metadata_objects:
            return

        full_metadata_objects: list[dict[str, Any]] = []
        for metadata_object in metadata_objects:
            if not isinstance(metadata_object, dict):
                continue
            full_object = dict(metadata_object)
            full_object["visible"] = True
            full_metadata_objects.append(full_object)

        if not full_metadata_objects:
            return

        full_observation = dict(observation)
        full_observation["request_id"] = request_id
        full_observation["scene_name"] = str(observation.get("scene_name") or fallback_scene_name or "")
        full_observation["metadata"] = {
            **(metadata if isinstance(metadata, dict) else {}),
            "objects": full_metadata_objects,
        }
        previous_scene_state_mode = getattr(self.tools, "scene_state_update_mode", None)
        previous_force_full_contents = getattr(
            self.tools,
            "force_full_observation_receptacle_contents",
            False,
        )
        try:
            self.tools.scene_state_update_mode = "metadata"
            self.tools.force_full_observation_receptacle_contents = True
            self.tools._sync_scene_state_from_observation(  # pylint: disable=protected-access
                full_observation,
                action_name="observe",
            )
        finally:
            if previous_scene_state_mode is not None:
                self.tools.scene_state_update_mode = previous_scene_state_mode
            self.tools.force_full_observation_receptacle_contents = previous_force_full_contents

    def _robot_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            deepcopy(tool_def)
            for tool_def in self.tools.get_definitions()
            if not str(tool_def.get("function", {}).get("name", "")).startswith("memory__")
        ]

    def _robot_tool_choice(self) -> str:
        return "required"

    def _tool_call_dicts_from_response(self, response) -> list[dict[str, Any]]:
        return [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                },
            }
            for tc in response.tool_calls
        ]

    def _dedupe_failure_messages(self, failure_messages: list[str]) -> list[str]:
        unique_failures: list[str] = []
        for message in failure_messages:
            normalized = str(message).strip()
            if normalized and normalized not in unique_failures:
                unique_failures.append(normalized)
        return unique_failures

    def _summarize_plan_failure(self, label: str, failure_messages: list[str]) -> str:
        unique_failures = self._dedupe_failure_messages(failure_messages)
        if not unique_failures:
            return f"{label} failed without a specific runtime error."
        return f"{label} failed: " + " | ".join(unique_failures)

    def _execute_robot_tool_calls(
        self,
        *,
        iteration: int,
        robot_tool_calls: list[Any],
        messages: list[dict[str, Any]],
        response_snapshot: dict[str, Any],
        response_meta_snapshot: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        failure_messages: list[str] = []

        if not robot_tool_calls:
            return messages, ["The model returned no robot action tool calls."]

        for tool_call in robot_tool_calls:
            self._log_tool_call(iteration, tool_call.name, tool_call.arguments)

        for tool_call in robot_tool_calls:
            batch_results = self.tools.execute_robot_batch_until_all_terminal(
                [(tool_call.name, tool_call.arguments)]
            )
            result_dict = batch_results[0] if batch_results else {
                "success": False,
                "feedback": "robot tool execution returned no result",
                "in_progress": False,
            }
            self._log_tool_result(iteration, tool_call.name, result_dict)
            self._record_unauthorized_tool_call(
                iteration=iteration,
                tool_name=tool_call.name,
                tool_args=tool_call.arguments,
                result_dict=result_dict,
                response_snapshot=response_snapshot,
                response_meta_snapshot=response_meta_snapshot,
            )
            self._save_robot_tool_image(
                iteration,
                tool_call.id,
                tool_call.name,
                tool_call.arguments,
                result_dict,
            )
            messages = self.context.add_tool_result(
                messages,
                tool_call.id,
                tool_call.name,
                result_dict,
            )
            if (not result_dict.get("success")) or result_dict.get("in_progress"):
                failure_messages.append(
                    f"{tool_call.name}: {result_dict.get('feedback') or 'robot tool execution failed'}"
                )

        return messages, failure_messages

    def process_request_messages(
        self,
        request,
        request_id: str | None = None,
        room_id: str | None = None,
        request_type: str | None = None,
        active_skill: str | None = None,
        request_injections: list[dict[str, Any]] | None = None,
    ):
        self.runtime_failure_reason = None

        if request_injections:
            injected_messages: list[str] = []
            for injection in request_injections:
                if not isinstance(injection, dict):
                    continue
                tasks = [dict(task) for task in (injection.get("tasks") or []) if isinstance(task, dict)]
                if tasks:
                    self.register_request_tasks(
                        tasks,
                        request_type=injection.get("request_type") or request_type,
                        active_skill=injection.get("active_skill") or active_skill,
                        replace=False,
                        preserve_robot_assignments=True,
                    )
                message_text = str(injection.get("message") or "").strip()
                if message_text:
                    injected_messages.append(message_text)

            if injected_messages:
                request = "\n\n".join([str(request).strip(), *injected_messages]).strip()

        return super().process_request_messages(
            request=request,
            request_id=request_id,
            room_id=room_id,
            request_type=request_type,
            active_skill=active_skill,
            request_injections=None,
        )

    def _run_agent_loop(
        self,
        initial_messages,
        active_skill: str | None = None,
        request_injections: list[dict[str, Any]] | None = None,
    ):
        del request_injections

        messages = initial_messages
        tools_used: list[str] = []
        llm_rounds: list[dict[str, Any]] = []
        iteration = int(getattr(self, "_baseline_round_offset", 0) or 0) + 1

        messages = self._refresh_system_prompt(messages, active_skill=active_skill)
        llm_messages = self._sanitize_messages_for_llm(messages)
        llm_input_snapshot = deepcopy(llm_messages)
        self._log_round_input(iteration, llm_input_snapshot)
        api_tools = self._robot_tool_definitions()
        tool_consistency_report = self._build_tool_consistency_report(llm_input_snapshot, api_tools)

        response = self.mllm.chat(
            messages=llm_messages,
            tools=api_tools,
            tool_choice=self._robot_tool_choice(),
            model=self.model,
        )

        response_snapshot = self._build_llm_response_snapshot(response)
        response_meta_snapshot = self._build_llm_response_meta_snapshot(response)
        round_log_payload = {
            "llm_input_messages": llm_input_snapshot,
            "tool_consistency": tool_consistency_report,
            "llm_response": deepcopy(response_snapshot),
            "llm_response_meta": deepcopy(response_meta_snapshot),
        }
        llm_rounds.append(
            {
                "iteration": iteration,
                "messages_sent_to_llm": llm_input_snapshot,
                "llm_response": deepcopy(response_snapshot),
                "llm_response_meta": deepcopy(response_meta_snapshot),
            }
        )
        self._log_llm_response(iteration, response_snapshot, response_meta_snapshot)

        final_content: str | None = None
        failure_messages: list[str] = []

        if not response.has_tool_calls:
            clean = self._strip_think(response.content)
            messages = self.context.add_assistant_message(
                messages,
                clean,
                reasoning_content=response.reasoning_content,
            )
            self._attach_llm_call_metadata_to_last_assistant_message(messages, response)
            self.runtime_failure_reason = (
                "Baseline-Viki planning failed: the model returned no tool calls in one-shot planning mode."
            )
            final_content = clean or self.runtime_failure_reason
            self._save_round_artifacts(iteration, messages, round_log_payload=round_log_payload)
            self._baseline_round_offset = iteration
            return final_content, tools_used, messages, llm_rounds

        tool_call_dicts = self._tool_call_dicts_from_response(response)
        messages = self.context.add_assistant_message(
            messages,
            response.content,
            tool_call_dicts,
            reasoning_content=response.reasoning_content,
        )
        self._attach_llm_call_metadata_to_last_assistant_message(messages, response)

        memory_tool_results: list[tuple[Any, dict[str, Any]]] = []
        robot_tool_calls: list[Any] = []
        for tool_call in response.tool_calls:
            tools_used.append(tool_call.name)
            if tool_call.name.startswith("memory__"):
                self._log_tool_call(iteration, tool_call.name, tool_call.arguments)
                result_dict = self.tools.execute(tool_call.name, tool_call.arguments)
                self._log_tool_result(iteration, tool_call.name, result_dict)
                memory_tool_results.append((tool_call, result_dict))
            else:
                robot_tool_calls.append(tool_call)

        for tool_call, result_dict in memory_tool_results:
            messages = self.context.add_tool_result(
                messages,
                tool_call.id,
                tool_call.name,
                result_dict,
            )
            if not result_dict.get("success"):
                failure_messages.append(
                    f"{tool_call.name}: {result_dict.get('feedback') or 'memory tool execution failed'}"
                )

        if not robot_tool_calls:
            failure_messages.append("The model returned no robot action tool calls.")
        else:
            messages, robot_failure_messages = self._execute_robot_tool_calls(
                iteration=iteration,
                robot_tool_calls=robot_tool_calls,
                messages=messages,
                response_snapshot=response_snapshot,
                response_meta_snapshot=response_meta_snapshot,
            )
            failure_messages.extend(robot_failure_messages)

        if failure_messages:
            self.runtime_failure_reason = self._summarize_plan_failure(
                "Baseline-Viki plan execution",
                failure_messages,
            )
            final_content = self.runtime_failure_reason
        else:
            final_content = (
                self._strip_think(response.content)
                or "Baseline-Viki plan executed without runtime tool failures."
            )

        self._save_round_artifacts(iteration, messages, round_log_payload=round_log_payload)
        self._baseline_round_offset = iteration
        return final_content, tools_used, messages, llm_rounds


class BaselineVikiReflectAgent(BaselineVikiAgent):
    """Baseline-Viki with up to three failure-feedback replanning opportunities."""

    max_reflection_rounds = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = BaselineVikiReflectContextBuilder(self.workspace)
        self.context.memory = self.memory
        self.context.set_tool_registry(self.tools)

    def _build_reflection_message(
        self,
        *,
        attempt: int,
        failure_messages: list[str],
    ) -> str:
        failure_summary = self._summarize_plan_failure(
            f"Attempt {attempt} execution",
            failure_messages,
        )
        memory_context = self.memory.get_memory_context()
        return "\n\n".join(
            [
                "The previous full-plan attempt produced runtime errors.",
                failure_summary,
                (
                    "Reflect on these errors and replan from the current environment and memory state. "
                    "Emit only the complete robot action tool-call sequence that should be run next. "
                    "Avoid repeating actions that already succeeded unless they are necessary to recover."
                ),
                f"Current memory after attempt {attempt}:\n{memory_context}" if memory_context else "",
            ]
        ).strip()

    def _run_agent_loop(
        self,
        initial_messages,
        active_skill: str | None = None,
        request_injections: list[dict[str, Any]] | None = None,
    ):
        del request_injections

        messages = initial_messages
        tools_used: list[str] = []
        llm_rounds: list[dict[str, Any]] = []
        final_content: str | None = None
        last_failure_messages: list[str] = []

        start_iteration = int(getattr(self, "_baseline_round_offset", 0) or 0) + 1
        max_iteration = start_iteration + int(self.max_reflection_rounds)
        for iteration in range(start_iteration, max_iteration + 1):
            messages = self._refresh_system_prompt(messages, active_skill=active_skill)
            llm_messages = self._sanitize_messages_for_llm(messages)
            llm_input_snapshot = deepcopy(llm_messages)
            self._log_round_input(iteration, llm_input_snapshot)
            api_tools = self._robot_tool_definitions()
            tool_consistency_report = self._build_tool_consistency_report(llm_input_snapshot, api_tools)

            response = self.mllm.chat(
                messages=llm_messages,
                tools=api_tools,
                tool_choice=self._robot_tool_choice(),
                model=self.model,
            )

            response_snapshot = self._build_llm_response_snapshot(response)
            response_meta_snapshot = self._build_llm_response_meta_snapshot(response)
            round_log_payload = {
                "llm_input_messages": llm_input_snapshot,
                "tool_consistency": tool_consistency_report,
                "llm_response": deepcopy(response_snapshot),
                "llm_response_meta": deepcopy(response_meta_snapshot),
            }
            llm_rounds.append(
                {
                    "iteration": iteration,
                    "messages_sent_to_llm": llm_input_snapshot,
                    "llm_response": deepcopy(response_snapshot),
                    "llm_response_meta": deepcopy(response_meta_snapshot),
                }
            )
            self._log_llm_response(iteration, response_snapshot, response_meta_snapshot)

            if not response.has_tool_calls:
                clean = self._strip_think(response.content)
                messages = self.context.add_assistant_message(
                    messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                )
                self._attach_llm_call_metadata_to_last_assistant_message(messages, response)
                last_failure_messages = [
                    "The model returned no robot action tool calls in one-shot planning mode."
                ]
                self._save_round_artifacts(iteration, messages, round_log_payload=round_log_payload)
            else:
                tool_call_dicts = self._tool_call_dicts_from_response(response)
                messages = self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                self._attach_llm_call_metadata_to_last_assistant_message(messages, response)

                robot_tool_calls = []
                memory_tool_failures: list[str] = []
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    if tool_call.name.startswith("memory__"):
                        result_dict = {
                            "tool": tool_call.name,
                            "success": False,
                            "feedback": "memory tools are unavailable in Baseline-Viki-Reflect",
                        }
                        self._log_tool_call(iteration, tool_call.name, tool_call.arguments)
                        self._log_tool_result(iteration, tool_call.name, result_dict)
                        messages = self.context.add_tool_result(
                            messages,
                            tool_call.id,
                            tool_call.name,
                            result_dict,
                        )
                        memory_tool_failures.append(f"{tool_call.name}: {result_dict['feedback']}")
                    else:
                        robot_tool_calls.append(tool_call)

                messages, robot_failure_messages = self._execute_robot_tool_calls(
                    iteration=iteration,
                    robot_tool_calls=robot_tool_calls,
                    messages=messages,
                    response_snapshot=response_snapshot,
                    response_meta_snapshot=response_meta_snapshot,
                )
                last_failure_messages = [*memory_tool_failures, *robot_failure_messages]
                self._save_round_artifacts(iteration, messages, round_log_payload=round_log_payload)

            if not last_failure_messages:
                self.runtime_failure_reason = None
                final_content = (
                    self._strip_think(response.content)
                    or f"Baseline-Viki-Reflect plan executed without runtime tool failures on attempt {iteration}."
                )
                break

            if iteration >= max_iteration:
                self.runtime_failure_reason = self._summarize_plan_failure(
                    "Baseline-Viki-Reflect plan execution",
                    last_failure_messages,
                )
                final_content = self.runtime_failure_reason
                break

            reflection_message = self._build_reflection_message(
                attempt=iteration,
                failure_messages=last_failure_messages,
            )
            messages.append(
                {
                    "role": "user",
                    "content": self.context.build_user_content(reflection_message, None),
                }
            )

        if llm_rounds:
            self._baseline_round_offset = int(llm_rounds[-1]["iteration"])
        return final_content, tools_used, messages, llm_rounds
