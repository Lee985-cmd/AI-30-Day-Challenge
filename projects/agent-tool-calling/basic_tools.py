"""
基础工具调用示例

展示如何使用LangChain的@tool装饰器创建和使用工具
"""

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


# ==================== 定义工具 ====================

@tool
def calculate(expression: str) -> str:
    """计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 2" 或 "12345 * 67890"
        
    Returns:
        计算结果字符串
    """
    try:
        # 安全计算（只允许基本运算）
        result = eval(expression, {"__builtins__": {}}, {})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """查询城市天气
    
    Args:
        city: 城市名称，如 "北京" 或 "上海"
        
    Returns:
        天气信息字符串
    """
    # 模拟天气API
    weather_data = {
        "北京": "晴，25°C，空气质量优",
        "上海": "多云，28°C，湿度65%",
        "广州": "小雨，30°C，湿度80%",
        "深圳": "晴，32°C，湿度70%",
        "杭州": "阴，26°C，湿度75%"
    }
    
    return weather_data.get(city, f"未找到{city}的天气信息")


@tool
def search_user(user_name: str) -> str:
    """搜索用户信息
    
    Args:
        user_name: 用户姓名
        
    Returns:
        用户信息字符串
    """
    # 模拟数据库
    database = {
        "张三": {"age": 30, "occupation": "工程师", "city": "北京"},
        "李四": {"age": 25, "occupation": "设计师", "city": "上海"},
        "王五": {"age": 35, "occupation": "产品经理", "city": "广州"}
    }
    
    if user_name in database:
        info = database[user_name]
        return f"{user_name}的信息：年龄{info['age']}岁，职业{info['occupation']}，所在城市{info['city']}"
    else:
        return f"未找到'{user_name}'的记录"


# ==================== 创建Agent ====================

def create_basic_agent():
    """创建基础Agent"""
    
    # 创建LLM
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        # 如果使用本地模型：
        # openai_api_base="http://localhost:8000/v1",
        # openai_api_key="local"
    )
    
    # 工具列表
    tools = [calculate, get_weather, search_user]
    
    # 提示词模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个有用的助手，可以使用各种工具来帮助用户。
        
        可用的工具：
        - calculate: 计算数学表达式
        - get_weather: 查询城市天气
        - search_user: 搜索用户信息
        
        当用户的问题需要这些信息时，请使用相应的工具。
        回答要简洁明了。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    # 创建Agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 创建执行器
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


# ==================== 使用示例 ====================

def example_basic_tools():
    """基础工具使用示例"""
    
    print("="*60)
    print("基础工具调用示例")
    print("="*60)
    
    # 创建Agent
    agent = create_basic_agent()
    
    # 测试用例
    test_cases = [
        "今天北京的天气怎么样？",
        "计算 12345 乘以 67890",
        "帮我查一下张三的信息",
        "上海的天气如何？",
        "计算 2的10次方"
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n[测试 {i}]")
        print(f"👤 用户: {query}")
        print("-" * 60)
        
        try:
            result = agent.invoke({"input": query})
            print(f"🤖 AI: {result['output']}")
        except Exception as e:
            print(f"❌ 错误: {e}")
        
        print("="*60)


if __name__ == "__main__":
    example_basic_tools()
