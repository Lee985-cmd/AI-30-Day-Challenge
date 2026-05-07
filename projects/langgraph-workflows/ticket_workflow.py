"""
智能客服工单系统 - LangGraph实现
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal
from enum import Enum
import json
from datetime import datetime

# 模拟LLM
class MockLLM:
    def invoke(self, prompt):
        return type('obj', (object,), {'content': '{"category": "billing", "urgency": "high"}'})()

llm = MockLLM()

# ==================== 状态定义 ====================

class TicketStatus(Enum):
    NEW = "new"
    IN_PROGRESS = "in_progress"
    NEEDS_REVIEW = "needs_review"
    RESOLVED = "resolved"
    ESCALATED = "escalated"

class TicketState(TypedDict):
    ticket_id: str
    customer_message: str
    category: str
    urgency: str
    knowledge_results: list
    draft_response: str
    needs_human_review: bool
    final_response: str
    status: TicketStatus
    history: list

# ==================== 节点函数 ====================

def classify_ticket(state: TicketState) -> dict:
    """分类工单"""
    print(f"📋 步骤1: 分类工单")
    
    message = state["customer_message"]
    
    # 模拟分类结果
    classification = {
        "category": "billing",
        "urgency": "high"
    }
    
    return {
        "category": classification["category"],
        "urgency": classification["urgency"],
        "status": TicketStatus.IN_PROGRESS,
        "history": [f"分类为: {classification['category']}, 紧急程度: {classification['urgency']}"]
    }

def retrieve_knowledge(state: TicketState) -> dict:
    """检索知识库"""
    print(f"🔍 步骤2: 检索知识库")
    
    # 模拟检索结果
    results = [
        "退款政策：7天内可全额退款",
        "账单问题处理流程",
        "常见问题解答"
    ]
    
    return {
        "knowledge_results": results,
        "history": state["history"] + [f"检索到 {len(results)} 条相关知识"]
    }

def generate_response(state: TicketState) -> dict:
    """生成回复"""
    print(f"✍️ 步骤3: 生成回复")
    
    response = """
尊敬的客户，

关于您的账单问题，我们已经收到并正在处理。

根据我们的退款政策，您可以在7天内申请全额退款。

我们将尽快为您处理，并在24小时内给出答复。

如有任何疑问，请随时联系我们。

祝您愉快！
    """
    
    # 判断是否需要人工审核
    needs_review = state["urgency"] in ["high", "critical"]
    
    return {
        "draft_response": response.strip(),
        "needs_human_review": needs_review,
        "status": TicketStatus.NEEDS_REVIEW if needs_review else TicketStatus.RESOLVED,
        "history": state["history"] + ["生成回复草稿"]
    }

def human_review_node(state: TicketState) -> dict:
    """人工审核节点"""
    print(f"👤 步骤4: 人工审核")
    
    # 模拟审核通过
    approved = True
    
    if approved:
        return {
            "final_response": state["draft_response"],
            "status": TicketStatus.RESOLVED,
            "history": state["history"] + ["人工审核通过"]
        }
    else:
        return {
            "status": TicketStatus.ESCALATED,
            "history": state["history"] + ["人工审核拒绝，已升级"]
        }

def send_response(state: TicketState) -> dict:
    """发送回复"""
    print(f"📤 步骤5: 发送回复")
    
    return {
        "history": state["history"] + ["回复已发送"]
    }

def log_ticket(state: TicketState) -> dict:
    """记录工单"""
    print(f"💾 步骤6: 记录工单")
    
    return {"history": state["history"] + ["工单已记录"]}

# ==================== 条件函数 ====================

def route_after_generation(state: TicketState) -> Literal["human_review", "send_response"]:
    """路由：生成回复后"""
    if state["needs_human_review"]:
        return "human_review"
    else:
        return "send_response"

def route_after_review(state: TicketState) -> Literal["send_response", "escalate"]:
    """路由：审核后"""
    if state["status"] == TicketStatus.RESOLVED:
        return "send_response"
    else:
        return "escalate"

# ==================== 构建图 ====================

def create_ticket_workflow():
    """创建工单处理工作流"""
    
    workflow = StateGraph(TicketState)
    
    # 添加节点
    workflow.add_node("classify", classify_ticket)
    workflow.add_node("retrieve", retrieve_knowledge)
    workflow.add_node("generate", generate_response)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("send", send_response)
    workflow.add_node("log", log_ticket)
    
    # 添加入口
    workflow.set_entry_point("classify")
    
    # 添加边
    workflow.add_edge("classify", "retrieve")
    workflow.add_edge("retrieve", "generate")
    
    # 条件边：生成后
    workflow.add_conditional_edges(
        "generate",
        route_after_generation,
        {
            "human_review": "human_review",
            "send_response": "send"
        }
    )
    
    # 条件边：审核后
    workflow.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "send_response": "send",
            "escalate": END
        }
    )
    
    # 发送后记录
    workflow.add_edge("send", "log")
    workflow.add_edge("log", END)
    
    return workflow.compile()

# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 80)
    print("智能客服工单系统 - LangGraph演示")
    print("=" * 80)
    print()
    
    # 创建工作流
    app = create_ticket_workflow()
    
    # 模拟工单
    initial_state = {
        "ticket_id": "TKT-2026-001",
        "customer_message": "我的账户被扣了两次费用，请帮我退款",
        "category": "",
        "urgency": "",
        "knowledge_results": [],
        "draft_response": "",
        "needs_human_review": False,
        "final_response": "",
        "status": TicketStatus.NEW,
        "history": []
    }
    
    print(f"工单ID: {initial_state['ticket_id']}")
    print(f"客户消息: {initial_state['customer_message']}")
    print()
    print("开始处理...")
    print("-" * 80)
    
    # 执行工作流
    result = app.invoke(initial_state)
    
    print("-" * 80)
    print()
    print("=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"\n工单ID: {result['ticket_id']}")
    print(f"状态: {result['status'].value}")
    print(f"分类: {result['category']}")
    print(f"紧急程度: {result['urgency']}")
    print(f"\n处理历史:")
    for i, step in enumerate(result['history'], 1):
        print(f"  {i}. {step}")
    print(f"\n最终回复:")
    print(result['final_response'])
    print("=" * 80)
