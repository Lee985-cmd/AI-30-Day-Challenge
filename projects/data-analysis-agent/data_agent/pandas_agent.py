"""
Pandas Agent - 用自然语言查询数据（基于 LangChain 代码生成）
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
import os
import pandas as pd

class PandasAIAgent:
    """Pandas AI Agent - 通过生成代码实现自然语言查询"""
    
    def __init__(self, api_key=None, base_url=None):
        # 使用本地模型
        self.base_url = base_url or os.getenv("LOCAL_LLM_URL", "")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "not-needed")
        
        # 初始化本地大模型（OpenAI 兼容接口）
        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model="qwen-plus",  # 根据实际模型名称调整
            temperature=0
        )
        
        # 数据分析提示词
        prompt = PromptTemplate(
            input_variables=["data_info", "question"],
            template="""你是一个数据分析专家。数据已经加载到变量 df 中（pandas DataFrame）。

数据信息：
{data_info}

问题：
{question}

请只输出 Python 代码，不要有任何解释、注释或markdown标记。代码要求：
1. 直接使用变量 df 进行数据分析，不要使用 pd.read_csv() 读取文件
2. 将分析结果赋值给变量 result
3. 如果结果是 DataFrame，转换为字符串：result = result.to_string()
4. 如果是数值或字符串，直接赋值

示例输出格式：
result = df.groupby('地区')['销售额'].sum()
result = result.idxmax()

你的代码："""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=prompt)
    
    def query(self, df: pd.DataFrame, question: str) -> dict:
        """
        用自然语言查询数据
        
        Args:
            df: 数据框
            question: 自然语言问题
            
        Returns:
            包含结果、代码、解释的字典
        """
        from data_loader import DataLoader
        data_info = DataLoader.get_data_info(df)
        
        # 生成分析代码
        result = self.chain.invoke({
            "data_info": data_info,
            "question": question
        })
        
        # 解析返回结果
        if isinstance(result, dict):
            # LangChain 可能返回 {'text': '...'} 或其他键
            code = result.get('text', result.get('output', str(result)))
        else:
            code = str(result)
        
        # 清理代码（去除可能的 markdown 标记）
        code = code.strip()
        if code.startswith('```python'):
            code = code[9:]
        if code.endswith('```'):
            code = code[:-3]
        code = code.strip()
        
        print(f"\n📝 生成的代码:\n{code}")
        
        # 执行代码
        try:
            # 创建执行环境
            exec_env = {"pd": pd, "df": df, "np": __import__('numpy')}
            
            # 在代码前后添加保障
            safe_code = f"result = None\n{code}\nif result is None:\n    result = '代码执行成功但未返回结果'"
            
            exec(safe_code, exec_env)
            
            answer = exec_env.get("result", "无结果")
            
            # 如果结果是 DataFrame，转换为字符串
            if isinstance(answer, pd.DataFrame):
                answer = answer.to_string()
            elif isinstance(answer, (int, float)):
                answer = str(answer)
            elif answer is None:
                answer = "无结果"
            
            print(f"\n✅ 执行成功，结果: {str(answer)[:200]}")
            
            return {
                "success": True,
                "result": str(answer),
                "explanation": f"问题：{question}\n\n回答：{answer}"
            }
        except Exception as e:
            print(f"\n❌ 执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "explanation": f"执行失败：{str(e)}\n\n生成的代码：\n{code}"
            }

# 测试
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # 加载数据
    df = DataLoader.load("../data/sales_data.xlsx")
    
    # 创建 Agent
    agent = PandasAIAgent()
    
    # 测试查询
    questions = [
        "哪个地区的销售额最高？",
        "Q3 各品类的销售额对比",
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        result = agent.query(df, q)
        print(result["explanation"])
