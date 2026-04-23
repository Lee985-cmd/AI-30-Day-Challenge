"""
AutoGPT 简化版 - 自主任务执行Agent
演示AI如何自主分解任务、执行、并完成任务目标
"""

from .autonomous_agent import AutonomousAgent
from .task_planner import TaskPlanner
from .executor import TaskExecutor
from .memory import ShortTermMemory, LongTermMemory

__version__ = "1.0.0"
__author__ = "Lee"

__all__ = [
    "AutonomousAgent",
    "TaskPlanner",
    "TaskExecutor",
    "ShortTermMemory",
    "LongTermMemory"
]
