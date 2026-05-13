"""
LangChain 完整示例
展示LangChain的核心功能：工具调用、记忆管理、链式编排
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from datetime import datetime

# 加载环境变量
load_dotenv()

# 配置LLM
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


def get_current_time():
    """获取当前时间"""
    return f"当前时间是: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def calculate(expression: str):
    """简单计算器"""
    try:
        # 注意：生产环境应使用更安全的计算方式
        result = eval(expression)
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def search_knowledge_base(query: str):
    """模拟知识库搜索"""
    knowledge = {
        "python": "Python是一种广泛使用的高级编程语言，强调代码可读性。",
        "ai": "人工智能(AI)是计算机科学的一个分支，致力于创建智能机器。",
        "langchain": "LangChain是一个用于开发LLM应用的框架。"
    }
    
    for key, value in knowledge.items():
        if key in query.lower():
            return value
    
    return "抱歉，没有找到相关信息。"


# ========== 示例1: 工具调用 Agent ==========
print("=" * 60)
print("示例1: LangChain 工具调用 Agent")
print("=" * 60)

# 定义工具
tools = [
    Tool(
        name="CurrentTime",
        func=get_current_time,
        description="获取当前日期和时间"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="执行数学计算，输入应该是数学表达式如 '2+2' 或 '10*5'"
    ),
    Tool(
        name="KnowledgeBase",
        func=search_knowledge_base,
        description="搜索知识库，输入应该是查询关键词"
    )
]

# 创建ReAct Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个有用的助手，可以使用工具来回答问题。
你可以使用以下工具：
{tools}

使用以下格式：
Question: 你必须回答的输入问题
Thought: 你应该总是思考该做什么
Action: 要采取的行动，应该是 [{tool_names}] 之一
Action Input: 行动的输入
Observation: 行动的结果
... (这个 Thought/Action/Action Input/Observation 可以重复 N 次)
Thought: 我现在知道最终答案
Final Answer: 对原始输入问题的最终答案"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 测试Agent
questions = [
    "现在几点了？",
    "计算 25 * 4 + 10",
    "什么是Python？",
    "先获取当前时间，然后告诉我Python是什么"
]

for question in questions:
    print(f"\n问题: {question}")
    try:
        response = agent_executor.invoke({"input": question})
        print(f"回答: {response['output']}")
    except Exception as e:
        print(f"错误: {str(e)}")


# ========== 示例2: 对话记忆 ==========
print("\n" + "=" * 60)
print("示例2: LangChain 对话记忆")
print("=" * 60)

memory = ConversationBufferMemory(return_messages=True)

conversation_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的聊天助手。记住之前的对话内容。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ]),
    memory=memory,
    verbose=True
)

conversations = [
    "我叫张三，今年25岁",
    "我住在北京",
    "你还记得我叫什么吗？",
    "我多大了？"
]

for message in conversations:
    print(f"\n用户: {message}")
    response = conversation_chain.run(input=message)
    print(f"助手: {response}")


# ========== 示例3: 链式编排 ==========
print("\n" + "=" * 60)
print("示例3: LangChain 链式编排")
print("=" * 60)

# 创建多个链并组合
summarize_prompt = ChatPromptTemplate.from_template(
    "请用一句话总结以下内容：\n{topic}"
)

translate_prompt = ChatPromptTemplate.from_template(
    "将以下内容翻译成英文：\n{text}"
)

summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt)

# 手动组合链
topic = "人工智能是模拟人类智能过程的计算机系统，包括学习、推理和自我修正等能力。"

print(f"\n原始内容: {topic}")
summary = summarize_chain.run(topic=topic)
print(f"摘要: {summary}")

translation = translate_chain.run(text=summary)
print(f"翻译: {translation}")


# ========== 示例4: 自定义工具 ==========
print("\n" + "=" * 60)
print("示例4: LangChain 自定义工具")
print("=" * 60)

class WeatherTool(Tool):
    """自定义天气查询工具"""
    
    def __init__(self):
        super().__init__(
            name="WeatherChecker",
            func=self.get_weather,
            description="查询城市天气，输入城市名称"
        )
    
    def get_weather(self, city: str):
        """模拟天气查询"""
        weather_data = {
            "北京": "晴朗，温度20°C",
            "上海": "多云，温度22°C",
            "广州": "小雨，温度25°C",
            "深圳": "晴天，温度26°C"
        }
        
        return weather_data.get(city, f"未找到 {city} 的天气信息")


weather_tool = WeatherTool()

# 创建带自定义工具的Agent
custom_tools = [weather_tool]
custom_agent = create_react_agent(llm, custom_tools, prompt)
custom_executor = AgentExecutor(agent=custom_agent, tools=custom_tools, verbose=False)

cities = ["北京", "上海", "成都"]
for city in cities:
    print(f"\n查询 {city} 天气:")
    try:
        response = custom_executor.invoke({"input": f"{city}的天气怎么样？"})
        print(f"结果: {response['output']}")
    except Exception as e:
        print(f"错误: {str(e)}")


print("\n" + "=" * 60)
print("LangChain 示例完成！")
print("=" * 60)
