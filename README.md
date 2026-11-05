# EmbodiedCoopBench

[GitHub 仓库](https://github.com/Meng-03/EmbodiedCoopBench) · [仓库修改计划表](DEVELOPMENT_PLAN.md)

当前处于开发阶段：60 个正式评测任务已独立提取；统一 agent 接口和完整运行入口仍在调整。后续修改及验收标准见计划表。

## Benchmark tasks

正式评测固定为 **60 个任务（episode / request bundle）**：酒店浴室清洁、设施巡检、餐厅服务、货物补给四类场景各 15 个，每类包含简单、中等、困难各 5 个。

一个评测任务由一个或多个请求组成；60 个任务共包含 180 次请求，引用 117 个去重后的请求实例。这些请求实例是任务内部的数据，不作为额外的评测任务计数。

- [任务说明及全部 60 个任务目录](tasks/README.md)
- [正式评测清单](tasks/manifest.json)：任务 ID、难度、场景、机器人配置和请求注入配置。
- `tasks/bundles/`：只包含正式评测的 60 个任务组合。
- `tasks/instances/`：只包含上述组合引用的初始化、目标判定和参考计划数据。

在仓库根目录运行 `python -m tasks.validate` 可以检查任务数量、引用完整性和难度分布，无需安装模拟器或配置模型。评测任务加载不依赖生成工具或候选任务池。

EmbodiedCoop 是一个面向 goal-driven 异构多机器人协作的 agent 框架与评测项目。它关注 interleaved task streams 中的协作问题：多个任务可能交错到达，场景状态并不完全可见，机器人动作反馈也是异步返回的。

EmbodiedCoop 通过事件驱动的 observe-update-dispatch 循环维护跨任务 progress memory，持续判断哪些目标项仍未知、需要执行、正在执行或已经完成，并据此调度具备不同能力的机器人。

## Overview

核心模块：

```text
agents/
  embodiedcoop/        # EmbodiedCoop agent 主体、memory、tool registry、LLM manager
  baseline-viki/       # Baseline-Viki agent
  baseline_roboos/     # Baseline-RoboOS agent

env/
  ai2thor/             # symbolic / multi-room 环境封装

eval/
  eval.py              # 主评测入口
  evalconfig/          # 评测配置
  setup/               # task 和 robot team 解析工具

tasks/
  manifest.json        # 固定的 60 个评测任务及场景配置
  bundles/             # 60 个任务组合，按场景和请求数量分文件
  instances/           # 仅保留组合所需的请求实例
  README.md            # 任务定义、数量和完整任务目录
  validate.py          # 不依赖模拟器的任务集完整性检查

robots/
  team_config/         # robot team presets

skills/
  */SKILL.md           # 不同场景的 agent skill prompt
```

主评测文件：

- `eval/eval.py`
- `eval/evalconfig/eval_4scenes_config.yaml`
- `tasks/manifest.json`

## Install

1. 创建并激活 Python 环境。
```bash
conda create -n embodiedcoop python=3.10
conda activate embodiedcoop
```

2. 安装依赖。
```bash
cd embodiedcoop
pip install -r requirements.txt
```


## Eval

1. 配置评测参数和 LLM 信息。
`eval/evalconfig/eval_4scenes_config.yaml` 用于控制评测入口参数，`agents/embodiedcoop/llm_manager/` 下保存模型和 API key 相关配置。

2. 确认场景信息
场景对应的 task 文件、robot preset、skill 名称等会从下面的文件自动读取：
    ```text
    tasks/manifest.json
    ```

3. 启动评测。
    ```bash
    python -m eval.eval \
    --config-file eval/evalconfig/eval_4scenes_config.yaml \
    --scene restocking \
    --bundle-id req3_restocking_sum03_bundle_000000 \
    --agent-mode embodiedcoop
    ```

该配置流程下，常用的核心字段包括：
- `agent_mode`
- `scene`
- `bundle_id`
- `model`
- `max_iterations`

### List Robot Teams

```bash
python -m eval.eval \
  --config-file eval/evalconfig/eval_4scenes_config.yaml \
  --list-robot-team-presets
```

### Available Scenes

```text
hotel_housekeeping
facility_inspection
restaurant_serving
restocking
```

每个场景可选的 bundle id 维护在：

```text
tasks/manifest.json
```

## Outputs

评测输出默认写入：

```text
logs/embodiedcoop/eval_runs/<model>/<category>/<method>/<scene>/
```

每次运行会保存：

- selected request bundle
- memory snapshot
- agent rounds
- environment images
- request evaluation summary
- timeline analysis artifacts
