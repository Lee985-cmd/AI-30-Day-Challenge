"""
基础ReAct Agent实现

从零实现ReAct框架，展示核心原理
"""

from typing import List, Dict
import re


class ReActAgent:
    """ReAct Agent实现"""
    
    def __init__(self, llm, tools: List[callable], max_steps: int = 10):
        self.llm = llm
        self.tools = {tool.__name__: tool for tool in tools}
        self.max_steps = max_steps
        
        # 历史记录
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.step_count = 0
    
    def run(self, question: str) -> str:
        """运行ReAct流程"""
        print(f"🤔 问题: {question}\n")
        
        for step in range(self.max_steps):
            self.step_count = step + 1
            print(f"--- Step {step + 1} ---")
            
            # Step 1: Thought - 生成思考
            thought = self._generate_thought(question)
            self.thoughts.append(thought)
            print(f"💭 Thought: {thought}")
            
            # Step 2: Action - 解析并执行行动
            action_name, action_input = self._parse_action(thought)
            
            if action_name == "Finish":
                final_answer = action_input
                print(f"✅ Final Answer: {final_answer}")
                return final_answer
            
            # 执行工具
            if action_name in self.tools:
                observation = self.tools[action_name](action_input)
                self.actions.append({"name": action_name, "input": action_input})
                self.observations.append(observation)
                print(f"🔧 Action: {action_name}({action_input})")
                print(f"👁️  Observation: {observation}\n")
            else:
                observation = f"未知工具: {action_name}"
                print(f"❌ {observation}\n")
        
        return "抱歉，我无法在限定步数内解决这个问题"
    
    def _generate_thought(self, question: str) -> str:
        """生成思考内容"""
        history = self._format_history()
        
        prompt = f"""你是一个智能助手，使用ReAct框架解决问题。

可用工具：
{self._format_tools()}

当前问题：{question}

{history}

请按照以下格式回答：
Thought: <你的思考>
Action: <工具名>[<参数>]

或者如果你已经知道答案：
Thought: 我已经知道了答案
Action: Finish[<最终答案>]
"""
        
        response = self.llm.invoke(prompt).content
        return response
    
    def _parse_action(self, thought: str) -> tuple:
        """解析行动"""
        action_match = re.search(r'Action:\s*(\w+)\[(.+?)\]', thought)
        
        if action_match:
            action_name = action_match.group(1)
            action_input = action_match.group(2)
            return action_name, action_input
        
        return "Finish", thought
    
    def _format_history(self) -> str:
        """格式化历史记录"""
        if not self.thoughts:
            return ""
        
        history_parts = []
        for i, (thought, obs) in enumerate(zip(self.thoughts, self.observations)):
            history_parts.append(f"Step {i+1}:")
            history_parts.append(f"Thought: {thought}")
            if i < len(self.actions):
                action = self.actions[i]
                history_parts.append(f"Action: {action['name']}[{action['input']}]")
            if obs:
                history_parts.append(f"Observation: {obs}")
            history_parts.append("")
        
        return "\n".join(history_parts)
    
    def _format_tools(self) -> str:
        """格式化工具列表"""
        tools_desc = []
        for name, tool in self.tools.items():
            doc = tool.__doc__ or "无描述"
            tools_desc.append(f"- {name}: {doc.split(chr(10))[0]}")
        return "\n".join(tools_desc)


# ==================== 工具定义 ====================

def search(query: str) -> str:
    """搜索网络信息"""
    knowledge_base = {
        "2023年诺贝尔文学奖得主": "Jon Fosse",
        "Jon Fosse 出生地": "Haugesund, Norway",
        "Haugesund 人口": "37,000",
        "Python最新版本": "Python 3.12",
        "北京天气": "晴，25°C"
    }
    
    return knowledge_base.get(query, f"未找到'{query}'的信息")


def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except:
        return "计算错误"


# ==================== 使用示例 ====================

def example_basic_react():
    """基础ReAct示例"""
    
    from langchain_openai import ChatOpenAI
    
    # 创建LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        # 本地模型配置：
        # openai_api_base="http://localhost:8000/v1",
        # openai_api_key="local"
    )
    
    # 创建Agent
    agent = ReActAgent(
        llm=llm,
        tools=[search, calculate],
        max_steps=10
    )
    
    # 测试问题
    question = "2023年诺贝尔文学奖得主的出生地人口是多少？"
    
    answer = agent.run(question)
    print(f"\n🎯 最终答案: {answer}")


if __name__ == "__main__":
    example_basic_react()
