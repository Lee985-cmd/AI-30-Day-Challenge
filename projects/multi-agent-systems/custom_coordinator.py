"""
自定义Agent协调器 - 不依赖CrewAI的轻量级实现

展示如何从零构建多Agent协作系统
"""

from typing import List, Dict, Any, Callable
from dataclasses import dataclass
import time


@dataclass
class Agent:
    """Agent定义"""
    name: str
    role: str
    expertise: str
    process_fn: Callable
    
    def process(self, task: str, context: Dict[str, Any] = None) -> str:
        """处理任务"""
        return self.process_fn(task, context or {})


class AgentCoordinator:
    """Agent协调器"""
    
    def __init__(self):
        self.agents: List[Agent] = []
        self.task_history: List[Dict[str, Any]] = []
        self.execution_log: List[Dict[str, Any]] = []
    
    def add_agent(self, agent: Agent):
        """添加Agent"""
        self.agents.append(agent)
        print(f"✅ 添加Agent: {agent.name} ({agent.role})")
    
    def assign_task(self, task: str, agent_name: str, 
                   context: Dict[str, Any] = None) -> str:
        """分配任务给指定Agent"""
        agent = next((a for a in self.agents if a.name == agent_name), None)
        
        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found")
        
        print(f"\n📋 分配任务给 {agent_name}:")
        print(f"   任务: {task[:100]}...")
        
        start_time = time.time()
        result = agent.process(task, context or {})
        duration = time.time() - start_time
        
        # 记录任务历史
        self.task_history.append({
            "task": task,
            "agent": agent_name,
            "result": result,
            "duration": duration
        })
        
        # 记录执行日志
        self.execution_log.append({
            "timestamp": time.time(),
            "agent": agent_name,
            "action": "completed",
            "duration": duration
        })
        
        print(f"✅ 完成 (耗时: {duration:.2f}s)")
        
        return result
    
    def sequential_execute(self, tasks: List[Dict[str, Any]]) -> List[str]:
        """顺序执行任务链
        
        Args:
            tasks: 任务列表，每个任务包含：
                - task: 任务描述
                - agent: Agent名称
        
        Returns:
            结果列表
        """
        print("\n" + "="*60)
        print("开始顺序执行任务链")
        print("="*60)
        
        results = []
        context = {}
        
        for i, task_info in enumerate(tasks):
            print(f"\n[步骤 {i+1}/{len(tasks)}]")
            
            # 将之前的结果加入上下文
            context["previous_results"] = results
            
            result = self.assign_task(
                task_info["task"],
                task_info["agent"],
                context
            )
            results.append(result)
            
            # 更新上下文
            context[f"result_{i}"] = result
        
        print("\n" + "="*60)
        print("✅ 所有任务完成！")
        print("="*60)
        
        return results
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        if not self.task_history:
            return {"error": "没有执行记录"}
        
        total_duration = sum(t["duration"] for t in self.task_history)
        
        return {
            "total_tasks": len(self.task_history),
            "total_duration": round(total_duration, 2),
            "avg_duration": round(total_duration / len(self.task_history), 2),
            "agents_used": list(set(t["agent"] for t in self.task_history))
        }
    
    def visualize_timeline(self):
        """可视化执行时间线"""
        print("\n📊 执行时间线:")
        print("-" * 60)
        
        for i, event in enumerate(self.task_history):
            bar_length = int(event["duration"] * 10)
            bar = "█" * bar_length
            print(f"{event['agent']:15s} | {bar} {event['duration']:.2f}s")


# ==================== 示例Agent实现 ====================

def create_research_agent():
    """创建研究员Agent"""
    def process(task: str, context: Dict[str, Any]) -> str:
        # 模拟研究过程
        time.sleep(0.5)
        return f"[研究报告] 针对'{task}'进行了深度调研，发现以下关键信息：\n1. 市场规模持续增长\n2. 技术创新加速\n3. 竞争格局变化"
    
    return Agent(
        name="researcher",
        role="研究员",
        expertise="市场调研、数据分析",
        process_fn=process
    )


def create_analyst_agent():
    """创建分析师Agent"""
    def process(task: str, context: Dict[str, Any]) -> str:
        # 模拟分析过程
        time.sleep(0.3)
        previous = context.get("previous_results", [])
        return f"[分析报告] 基于{len(previous)}个前期研究结果进行分析：\n- 优势：技术领先、市场认可\n- 劣势：成本高、人才短缺\n- 机会：政策支持、需求增长\n- 威胁：竞争加剧、技术迭代"
    
    return Agent(
        name="analyst",
        role="分析师",
        expertise="财务分析、风险评估",
        process_fn=process
    )


def create_writer_agent():
    """创建作家Agent"""
    def process(task: str, context: Dict[str, Any]) -> str:
        # 模拟写作过程
        time.sleep(0.4)
        previous = context.get("previous_results", [])
        return f"[最终报告] 综合{len(previous)}份前期材料，撰写完成：\n\n# 行业分析报告\n\n## 摘要\n本报告深入分析了当前行业发展态势...\n\n## 结论\n建议积极布局，把握发展机遇。"
    
    return Agent(
        name="writer",
        role="作家",
        expertise="报告撰写、内容创作",
        process_fn=process
    )


# ==================== 主函数 ====================

def main():
    """主函数"""
    print("🤖 自定义Agent协调器示例\n")
    
    # 创建协调器
    coordinator = AgentCoordinator()
    
    # 添加Agent
    coordinator.add_agent(create_research_agent())
    coordinator.add_agent(create_analyst_agent())
    coordinator.add_agent(create_writer_agent())
    
    # 定义任务链
    tasks = [
        {"task": "人工智能行业发展现状", "agent": "researcher"},
        {"task": "SWOT分析", "agent": "analyst"},
        {"task": "撰写行业分析报告", "agent": "writer"}
    ]
    
    # 执行任务链
    results = coordinator.sequential_execute(tasks)
    
    # 输出结果
    print("\n" + "="*60)
    print("📄 最终输出:")
    print("="*60)
    print(results[-1])
    
    # 显示执行摘要
    summary = coordinator.get_execution_summary()
    print("\n" + "="*60)
    print("📊 执行摘要:")
    print("="*60)
    print(f"总任务数: {summary['total_tasks']}")
    print(f"总耗时: {summary['total_duration']}s")
    print(f"平均耗时: {summary['avg_duration']}s")
    print(f"使用的Agent: {', '.join(summary['agents_used'])}")
    
    # 可视化时间线
    coordinator.visualize_timeline()


if __name__ == "__main__":
    main()
