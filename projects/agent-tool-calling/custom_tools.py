"""
自定义工具类示例

展示如何使用BaseTool创建高级工具
"""

from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


# ==================== 定义输入Schema ====================

class WeatherInput(BaseModel):
    """天气查询输入"""
    city: str = Field(description="城市名称，如'北京'、'上海'")
    date: str = Field(description="日期，格式YYYY-MM-DD，默认为今天", default="today")


class CalculatorInput(BaseModel):
    """计算器输入"""
    expression: str = Field(description="数学表达式，如 '2 + 2' 或 '12345 * 67890'")


# ==================== 自定义工具类 ====================

class WeatherTool(BaseTool):
    """天气查询工具"""
    
    name: str = "get_weather"
    description: str = "查询指定城市的天气信息，返回温度、天气状况和湿度"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, city: str, date: str = "today") -> str:
        """同步执行"""
        # 模拟天气API
        weather_data = {
            "北京": {"temp": 25, "condition": "晴", "humidity": 40},
            "上海": {"temp": 28, "condition": "多云", "humidity": 65},
            "广州": {"temp": 30, "condition": "小雨", "humidity": 80},
            "深圳": {"temp": 32, "condition": "晴", "humidity": 70},
            "杭州": {"temp": 26, "condition": "阴", "humidity": 75}
        }
        
        if city in weather_data:
            data = weather_data[city]
            return f"{city}{date}天气：{data['condition']}，温度{data['temp']}°C，湿度{data['humidity']}%"
        else:
            return f"未找到{city}的天气信息"
    
    async def _arun(self, city: str, date: str = "today") -> str:
        """异步执行"""
        return self._run(city, date)


class CalculatorTool(BaseTool):
    """计算器工具"""
    
    name: str = "calculate"
    description: str = "计算数学表达式，支持加减乘除和幂运算"
    args_schema: Type[BaseModel] = CalculatorInput
    
    def _run(self, expression: str) -> str:
        """执行计算"""
        try:
            # 安全计算（只允许基本运算）
            result = eval(expression, {"__builtins__": {}}, {})
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"
    
    async def _arun(self, expression: str) -> str:
        """异步执行"""
        return self._run(expression)


class UserInfoTool(BaseTool):
    """用户信息查询工具"""
    
    name: str = "search_user"
    description: str = "查询用户的个人信息，包括年龄、职业和所在城市"
    
    def _run(self, user_name: str) -> str:
        """查询用户信息"""
        # 模拟数据库
        database = {
            "张三": {"age": 30, "occupation": "工程师", "city": "北京", "salary": 20000},
            "李四": {"age": 25, "occupation": "设计师", "city": "上海", "salary": 15000},
            "王五": {"age": 35, "occupation": "产品经理", "city": "广州", "salary": 25000}
        }
        
        if user_name in database:
            info = database[user_name]
            return (f"用户{user_name}的信息：\n"
                   f"- 年龄：{info['age']}岁\n"
                   f"- 职业：{info['occupation']}\n"
                   f"- 城市：{info['city']}\n"
                   f"- 薪资：{info['salary']}元")
        else:
            return f"未找到用户'{user_name}'的记录"
    
    async def _arun(self, user_name: str) -> str:
        """异步执行"""
        return self._run(user_name)


# ==================== 使用示例 ====================

def example_custom_tools():
    """自定义工具使用示例"""
    
    print("="*60)
    print("自定义工具类示例")
    print("="*60)
    
    # 创建工具实例
    weather_tool = WeatherTool()
    calculator_tool = CalculatorTool()
    user_tool = UserInfoTool()
    
    # 测试工具
    print("\n🌤️  测试天气工具:")
    print(weather_tool.run({"city": "北京"}))
    print(weather_tool.run({"city": "上海"}))
    
    print("\n🔢 测试计算器工具:")
    print(calculator_tool.run({"expression": "2 + 2"}))
    print(calculator_tool.run({"expression": "12345 * 67890"}))
    print(calculator_tool.run({"expression": "2 ** 10"}))
    
    print("\n👤 测试用户查询工具:")
    print(user_tool.run({"user_name": "张三"}))
    print(user_tool.run({"user_name": "李四"}))
    
    # 查看工具信息
    print("\n📋 工具信息:")
    print(f"天气工具名称: {weather_tool.name}")
    print(f"天气工具描述: {weather_tool.description}")
    print(f"\n计算器工具名称: {calculator_tool.name}")
    print(f"计算器工具描述: {calculator_tool.description}")


if __name__ == "__main__":
    example_custom_tools()
