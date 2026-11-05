# agents/baseagent/base_agent.py

class BaseAgent:
    def __init__(self):
        self.plan =[]
        self.step_idx = 0
        self.current_task_id = None

    def reset(self, task_instance):
        """
        每次开始新任务前，重置 Agent 状态并加载黄金计划 (Ground Truth Plan)
        """
        self.current_task_id = task_instance["instance_id"]
        self.plan = task_instance.get("pddl_plan",[])
        self.step_idx = 0
        print(f"🤖 [BaseAgent] 已加载任务 {self.current_task_id}, 计划长度: {len(self.plan)}步")

    def step(self, observation=None):
        """
        根据当前进度输出下一个动作。
        由于是 BaseAgent(Oracle)，我们暂时忽略 observation(视觉/状态输入)，直接按顺序输出。
        """
        if self.step_idx < len(self.plan):
            action = self.plan[self.step_idx]
            self.step_idx += 1
            return action
        else:
            # 计划执行完毕，输出一个特殊的 Done 动作，告诉环境任务结束
            return {"action": "Done"}