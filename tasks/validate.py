"""Validate the fixed benchmark using only the Python standard library.

Run from the repository root: python -m tasks.validate
"""

from collections import Counter
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCENES = {
    "hotel_housekeeping": 40,
    "facility_inspection": 15,
    "restaurant_serving": 45,
    "restocking": 17,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root=REPO_ROOT):
    root = Path(root).resolve()
    manifest = read_json(root / "tasks/manifest.json")
    require(manifest["task_count"] == 60, "The official benchmark must have 60 tasks")
    require(manifest["evaluation_unit"] == "bundle", "Evaluation unit must be bundle")
    require(set(manifest["scenes"]) == set(EXPECTED_SCENES), "Unexpected scenes")
    referenced_files = {root / "tasks/manifest.json"}
    all_bundle_ids = set()
    rows = []

    def dataset_path(relative):
        path = (root / relative).resolve()
        require(path.is_relative_to(root / "tasks"), f"Data outside tasks/: {relative}")
        return path

    for scene, config in manifest["scenes"].items():
        entries = config["bundles"]
        require(len(entries) == 15, f"{scene}: expected 15 tasks")
        require(Counter(e["difficulty"] for e in entries) == {"low": 5, "mid": 5, "high": 5},
                f"{scene}: expected five tasks per difficulty")
        instance_path = dataset_path(config["task_json"])
        referenced_files.add(instance_path)
        by_room = read_json(instance_path)
        instances = {}
        for room, tasks in by_room.items():
            for task in tasks:
                task_id = task["instance_id"]
                require(task_id not in instances, f"Duplicate instance: {task_id}")
                require(task["scene_name"] == room, f"Wrong room for {task_id}")
                for field in ("init_setup", "init_setup_actions", "goal_verification", "pddl_plan"):
                    require(field in task, f"{task_id}: missing {field}")
                require("layout_path" not in task, f"{task_id}: stale generation path")
                instances[task_id] = task

        stored_bundles = {}
        directory = dataset_path(config["request_bundle_dir"])
        for count in sorted({e["request_bundle_count"] for e in entries}):
            path = directory / f"{config['request_bundle_file_prefix']}_req{count}.json"
            referenced_files.add(path)
            payload = read_json(path)
            require(payload["request_count"] == count, f"Wrong request count: {path}")
            for bundle in payload["bundles"]:
                bundle_id = bundle["bundle_id"]
                require(bundle_id not in stored_bundles, f"Duplicate bundle: {bundle_id}")
                require(bundle["request_count"] == count, f"Wrong file for {bundle_id}")
                stored_bundles[bundle_id] = bundle
        require(set(stored_bundles) == {e["bundle_id"] for e in entries},
                f"{scene}: missing or extra bundles")

        used_instances = set()
        request_total = 0
        for entry in entries:
            bundle_id = entry["bundle_id"]
            require(bundle_id not in all_bundle_ids, f"Duplicate task ID: {bundle_id}")
            all_bundle_ids.add(bundle_id)
            bundle = stored_bundles[bundle_id]
            requests = bundle["requests"]
            count = entry["request_bundle_count"]
            require(len(requests) == count == len(entry["tasks"]), f"{bundle_id}: request count mismatch")
            require(bundle["request_ids"] == [r["request_id"] for r in requests],
                    f"{bundle_id}: request IDs/order mismatch")
            require(len(set(bundle["request_ids"])) == count, f"{bundle_id}: duplicate requests")
            for field in ("bundle_total_active_objects", "bundle_total_pddl_plan_length"):
                require(bundle[field] == entry[field], f"{bundle_id}: {field} mismatch")
            for request, metadata in zip(requests, entry["tasks"]):
                for field in ("request_id", "instance_id", "scene_name", "room_id"):
                    require(request[field] == metadata[field], f"{bundle_id}: {field} mismatch")
                task_id = request["instance_id"]
                require(task_id in instances, f"Missing instance: {task_id}")
                require(instances[task_id]["scene_name"] == request["scene_name"],
                        f"{bundle_id}: incorrect instance room")
                used_instances.add(task_id)
            request_total += count
        require(used_instances == set(instances), f"{scene}: extra or missing request instances")
        require(len(instances) == EXPECTED_SCENES[scene], f"{scene}: unexpected instance count")
        require(request_total == 45, f"{scene}: expected 45 request occurrences")
        rows.append((scene, len(entries), request_total, len(instances)))

    require(len(all_bundle_ids) == 60, "Expected 60 unique task IDs")
    require(set((root / "tasks").rglob("*.json")) == referenced_files,
            "Unreferenced or missing JSON files in tasks/")
    return rows


def main():
    for scene, count, requests, instances in validate():
        print(f"{scene}: {count} tasks, 5/5/5 difficulty, {requests} requests, {instances} unique instances")
    print("PASS: 60 evaluation tasks, 180 request occurrences, 117 unique request instances")


if __name__ == "__main__":
    main()
