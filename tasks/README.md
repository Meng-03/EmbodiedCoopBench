# EmbodiedCoop 正式评测任务

正式评测集固定为 **60 个任务**。这里的一个“任务”指一个完整的评测 episode（代码中称 `request bundle`），可以包含一个或多个依次到达的请求。参与者应运行清单中的全部 60 个任务。

| 场景类别 | 任务数 | 简单（low） | 中等（mid） | 困难（high） | 请求出现次数 |
|---|---:|---:|---:|---:|---:|
| 酒店浴室清洁 | 15 | 5 | 5 | 5 | 45 |
| 设施巡检 | 15 | 5 | 5 | 5 | 45 |
| 餐厅服务 | 15 | 5 | 5 | 5 | 45 |
| 货物补给 | 15 | 5 | 5 | 5 | 45 |
| **合计** | **60** | **20** | **20** | **20** | **180** |

180 次请求引用 117 个去重后的请求实例。一个实例可能在不同任务组合中复用；它们是 60 个任务的组成部分，不是另外 117 个评测任务。四类场景是应用类别，不是四个模拟器房间。任务 ID 和难度沿用原主评测清单，没有重新采样。

## 任务内容

- **酒店浴室清洁**：按照[清洁规则](../skills/hotel_bathroom_housekeeping/SKILL.md)，检查房间状态并完成适用的清洁、整理与补充工作。
- **设施巡检**：按照[巡检规则](../skills/facility_inspection/SKILL.md)，检查并关闭笔记本、电视和灯具；电视操作还涉及遥控器取用与归位。
- **餐厅服务**：按照[服务规则](../skills/restaurant_serving/SKILL.md)，完成请求指定数量的煎蛋和吐司制作与上餐。
- **货物补给**：按照[补给规则](../skills/restocking/SKILL.md)，将鸡蛋、面包、苹果补足到指定位置与数量，并关闭使用过的储物设施。

每个请求的实际初始状态由 `init_setup` / `init_setup_actions` 定义，具体完成条件由 `goal_verification` 定义。上述描述帮助理解任务；最终计分使用评测器的目标判定逻辑。

## 数据组织

```text
tasks/
├── manifest.json                 # 60 个任务的权威清单及场景运行配置
├── bundles/<scene>/bundles_reqN.json
│                                 # 仅包含入选任务的请求顺序与组合信息
├── instances/<scene>.json        # 仅包含这些任务所需的请求实例数据
├── validate.py                   # 固定任务集完整性检查
└── README.md                     # 说明和全部任务目录
```

为兼容现有评测加载器，组合数据按场景与请求数量分文件存放，**JSON 文件数量不等于任务数量**。以 `manifest.json` 中的 60 个 `bundle_id` 为准。

- `manifest.json`：保存场景、难度、机器人预设、请求注入配置，以及每个任务引用的请求信息。
- `bundles/`：保留选中组合的原始 ID、请求顺序、组合文本和运行相关字段；不包含未入选组合或采样统计。
- `instances/`：保留必要的初始化、对象映射、目标判定与参考计划；不包含未被引用的候选实例。历史生成来源 `layout_path` 已移除，环境直接使用内嵌的初始化动作。
- `pddl_plan` 是参考计划，当前运行预算和 Oracle 可能使用它。保留该字段是为了维持现有任务语义；它不是要求普通 agent 使用的输入。
- 运行时场景目标等价规则放在 `eval/setup/scene_special_config.py`。这些数据与加载代码不依赖 `gen/`。

本次只提取任务数据并迁移路径，没有改变任务目标、机器人预设、难度或请求到达配置。参考计划和评测专用信息如何与 agent 隔离，将由后续 agent 接口调整处理。

## 检查与选择任务

在仓库根目录执行（仅需 Python 3.10+ 标准库）：

```bash
python -m tasks.validate
```

检查内容包括：60 个唯一任务 ID、每场景 15 个任务、各难度 5 个、请求顺序与数量、实例引用完整性，以及无额外候选实例或 JSON 文件。

在现有评测配置中通过 `scene` 和 `bundle_id` 选择下表中的任务；`eval_bundles_file` 已改为 `tasks/manifest.json`。任务数据可独立校验，完整模拟器评测仍需补齐当前缺失的 agent 实现。

## 全部 60 个任务

下表中的“请求数”是一个任务内部的请求数量；“活动对象数”和“参考计划长度”沿用清单统计，不代表额外任务数。房间按请求出现顺序列出，允许重复。

### 酒店浴室清洁（`hotel_housekeeping`）

| 任务 ID | 难度 | 请求数 | 房间 | 活动对象数 | 参考计划长度 |
|---|---|---:|---|---:|---:|
| [req1_bathroom_FloorPlan402_1439](bundles/hotel_housekeeping/bundles_req1.json) | low | 1 | FloorPlan402 | 1 | 5 |
| [req1_bathroom_FloorPlan402_1221](bundles/hotel_housekeeping/bundles_req1.json) | low | 1 | FloorPlan402 | 4 | 18 |
| [req1_bathroom_FloorPlan405_1175](bundles/hotel_housekeeping/bundles_req1.json) | low | 1 | FloorPlan405 | 5 | 21 |
| [req2_sum7_bundle_000001](bundles/hotel_housekeeping/bundles_req2.json) | low | 2 | FloorPlan405, FloorPlan406 | 7 | 32 |
| [req2_sum8_bundle_000005](bundles/hotel_housekeeping/bundles_req2.json) | low | 2 | FloorPlan413, FloorPlan414 | 8 | 34 |
| [req2_sum10_bundle_000003](bundles/hotel_housekeeping/bundles_req2.json) | mid | 2 | FloorPlan408, FloorPlan406 | 10 | 43 |
| [req3_sum11_bundle_000003](bundles/hotel_housekeeping/bundles_req3.json) | mid | 3 | FloorPlan403, FloorPlan413, FloorPlan405 | 11 | 50 |
| [req3_sum12_bundle_000001](bundles/hotel_housekeeping/bundles_req3.json) | mid | 3 | FloorPlan408, FloorPlan414, FloorPlan413 | 12 | 53 |
| [req3_sum14_bundle_000009](bundles/hotel_housekeeping/bundles_req3.json) | mid | 3 | FloorPlan405, FloorPlan414, FloorPlan408 | 14 | 60 |
| [req4_sum16_bundle_000019](bundles/hotel_housekeeping/bundles_req4.json) | mid | 4 | FloorPlan408, FloorPlan405, FloorPlan406, FloorPlan414 | 16 | 68 |
| [req4_sum18_bundle_000003](bundles/hotel_housekeeping/bundles_req4.json) | high | 4 | FloorPlan408, FloorPlan403, FloorPlan405, FloorPlan406 | 18 | 77 |
| [req4_sum20_bundle_000001](bundles/hotel_housekeeping/bundles_req4.json) | high | 4 | FloorPlan408, FloorPlan405, FloorPlan402, FloorPlan414 | 20 | 86 |
| [req5_sum22_bundle_000016](bundles/hotel_housekeeping/bundles_req5.json) | high | 5 | FloorPlan403, FloorPlan413, FloorPlan402, FloorPlan406, FloorPlan405 | 22 | 94 |
| [req5_sum24_bundle_000011](bundles/hotel_housekeeping/bundles_req5.json) | high | 5 | FloorPlan408, FloorPlan403, FloorPlan413, FloorPlan405, FloorPlan414 | 24 | 100 |
| [req5_sum28_bundle_000007](bundles/hotel_housekeeping/bundles_req5.json) | high | 5 | FloorPlan408, FloorPlan414, FloorPlan406, FloorPlan413, FloorPlan405 | 28 | 124 |

### 设施巡检（`facility_inspection`）

| 任务 ID | 难度 | 请求数 | 房间 | 活动对象数 | 参考计划长度 |
|---|---|---:|---|---:|---:|
| [req3_facility_inspection_bundle_000000](bundles/facility_inspection/bundles_req3.json) | low | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 3 | 6 |
| [req3_facility_inspection_bundle_000001](bundles/facility_inspection/bundles_req3.json) | low | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 4 | 8 |
| [req3_facility_inspection_bundle_000015](bundles/facility_inspection/bundles_req3.json) | low | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 4 | 8 |
| [req3_facility_inspection_bundle_000002](bundles/facility_inspection/bundles_req3.json) | low | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 5 | 10 |
| [req3_facility_inspection_bundle_000003](bundles/facility_inspection/bundles_req3.json) | low | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 6 | 12 |
| [req3_facility_inspection_bundle_000004](bundles/facility_inspection/bundles_req3.json) | mid | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 7 | 14 |
| [req3_facility_inspection_bundle_000019](bundles/facility_inspection/bundles_req3.json) | mid | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 8 | 16 |
| [req3_facility_inspection_bundle_000034](bundles/facility_inspection/bundles_req3.json) | mid | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 9 | 18 |
| [req3_facility_inspection_bundle_000049](bundles/facility_inspection/bundles_req3.json) | mid | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 10 | 20 |
| [req3_facility_inspection_bundle_000064](bundles/facility_inspection/bundles_req3.json) | mid | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 11 | 22 |
| [req3_facility_inspection_bundle_000214](bundles/facility_inspection/bundles_req3.json) | high | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 12 | 24 |
| [req3_facility_inspection_bundle_000364](bundles/facility_inspection/bundles_req3.json) | high | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 13 | 26 |
| [req3_facility_inspection_bundle_000514](bundles/facility_inspection/bundles_req3.json) | high | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 14 | 28 |
| [req3_facility_inspection_bundle_000663](bundles/facility_inspection/bundles_req3.json) | high | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 14 | 28 |
| [req3_facility_inspection_bundle_000664](bundles/facility_inspection/bundles_req3.json) | high | 3 | FloorPlan201, FloorPlan205, FloorPlan210 | 15 | 30 |

### 餐厅服务（`restaurant_serving`）

| 任务 ID | 难度 | 请求数 | 房间 | 活动对象数 | 参考计划长度 |
|---|---|---:|---|---:|---:|
| [req1_restaurant_serving_sum01_bundle_000000](bundles/restaurant_serving/bundles_req1.json) | low | 1 | FloorPlan11 | 1 | 6 |
| [req2_restaurant_serving_sum02_bundle_000000](bundles/restaurant_serving/bundles_req2.json) | low | 2 | FloorPlan18, FloorPlan18 | 2 | 12 |
| [req3_restaurant_serving_sum03_bundle_000000](bundles/restaurant_serving/bundles_req3.json) | low | 3 | FloorPlan4, FloorPlan4, FloorPlan4 | 3 | 18 |
| [req4_restaurant_serving_sum04_bundle_000000](bundles/restaurant_serving/bundles_req4.json) | low | 4 | FloorPlan16, FloorPlan16, FloorPlan16, FloorPlan16 | 4 | 24 |
| [req5_restaurant_serving_sum05_bundle_000000](bundles/restaurant_serving/bundles_req5.json) | low | 5 | FloorPlan28, FloorPlan28, FloorPlan28, FloorPlan28, FloorPlan28 | 5 | 30 |
| [req1_restaurant_serving_sum06_bundle_000000](bundles/restaurant_serving/bundles_req1.json) | mid | 1 | FloorPlan11 | 6 | 36 |
| [req2_restaurant_serving_sum07_bundle_000000](bundles/restaurant_serving/bundles_req2.json) | mid | 2 | FloorPlan18, FloorPlan18 | 7 | 42 |
| [req3_restaurant_serving_sum08_bundle_000000](bundles/restaurant_serving/bundles_req3.json) | mid | 3 | FloorPlan4, FloorPlan4, FloorPlan4 | 8 | 48 |
| [req1_restaurant_serving_sum09_bundle_000000](bundles/restaurant_serving/bundles_req1.json) | mid | 1 | FloorPlan28 | 9 | 54 |
| [req2_restaurant_serving_sum10_bundle_000000](bundles/restaurant_serving/bundles_req2.json) | mid | 2 | FloorPlan11, FloorPlan11 | 10 | 60 |
| [req3_restaurant_serving_sum11_bundle_000000](bundles/restaurant_serving/bundles_req3.json) | high | 3 | FloorPlan18, FloorPlan18, FloorPlan18 | 11 | 66 |
| [req4_restaurant_serving_sum12_bundle_000000](bundles/restaurant_serving/bundles_req4.json) | high | 4 | FloorPlan4, FloorPlan4, FloorPlan4, FloorPlan4 | 12 | 72 |
| [req5_restaurant_serving_sum13_bundle_000000](bundles/restaurant_serving/bundles_req5.json) | high | 5 | FloorPlan16, FloorPlan16, FloorPlan16, FloorPlan16, FloorPlan16 | 13 | 78 |
| [req5_restaurant_serving_sum14_bundle_000000](bundles/restaurant_serving/bundles_req5.json) | high | 5 | FloorPlan18, FloorPlan18, FloorPlan18, FloorPlan18, FloorPlan18 | 14 | 84 |
| [req4_restaurant_serving_sum15_bundle_000000](bundles/restaurant_serving/bundles_req4.json) | high | 4 | FloorPlan4, FloorPlan4, FloorPlan4, FloorPlan4 | 15 | 90 |

### 货物补给（`restocking`）

| 任务 ID | 难度 | 请求数 | 房间 | 活动对象数 | 参考计划长度 |
|---|---|---:|---|---:|---:|
| [req3_restocking_sum03_bundle_000000](bundles/restocking/bundles_req3.json) | low | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 3 | 21 |
| [req3_restocking_sum04_bundle_000000](bundles/restocking/bundles_req3.json) | low | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 4 | 25 |
| [req3_restocking_sum05_bundle_000000](bundles/restocking/bundles_req3.json) | low | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 5 | 29 |
| [req3_restocking_sum06_bundle_000000](bundles/restocking/bundles_req3.json) | low | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 6 | 33 |
| [req3_restocking_sum07_bundle_000000](bundles/restocking/bundles_req3.json) | low | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 7 | 37 |
| [req3_restocking_sum08_bundle_000000](bundles/restocking/bundles_req3.json) | mid | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 8 | 41 |
| [req3_restocking_sum09_bundle_000000](bundles/restocking/bundles_req3.json) | mid | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 9 | 45 |
| [req3_restocking_sum10_bundle_000000](bundles/restocking/bundles_req3.json) | mid | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 10 | 49 |
| [req3_restocking_sum11_bundle_000000](bundles/restocking/bundles_req3.json) | mid | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 11 | 53 |
| [req3_restocking_sum12_bundle_000000](bundles/restocking/bundles_req3.json) | mid | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 12 | 57 |
| [req3_restocking_sum13_bundle_000000](bundles/restocking/bundles_req3.json) | high | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 13 | 61 |
| [req3_restocking_sum14_bundle_000000](bundles/restocking/bundles_req3.json) | high | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 14 | 65 |
| [req3_restocking_sum15_bundle_000000](bundles/restocking/bundles_req3.json) | high | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 15 | 69 |
| [req3_restocking_sum16_bundle_000000](bundles/restocking/bundles_req3.json) | high | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 16 | 73 |
| [req3_restocking_sum17_bundle_000000](bundles/restocking/bundles_req3.json) | high | 3 | FloorPlan28, FloorPlan4, FloorPlan11 | 17 | 77 |

