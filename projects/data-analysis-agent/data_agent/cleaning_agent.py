"""
数据清洗 Agent
"""
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
import os

class DataCleaningAgent:
    """数据清洗 Agent"""
    
    def __init__(self, api_key=None, base_url=None):
        # 使用本地模型
        self.base_url = base_url or os.getenv("LOCAL_LLM_URL", "")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "not-needed")
        
        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model="qwen-plus",
            temperature=0
        )
        
        # 数据清洗提示词
        prompt = PromptTemplate(
            input_variables=["data_info", "cleaning_request"],
            template="""你是一个数据分析专家。请根据以下数据信息和清洗需求，生成 Python 代码。

数据信息：
{data_info}

清洗需求：
{cleaning_request}

请只输出 Python 代码，不要有任何解释、注释或markdown标记。代码应该：
1. 处理缺失值（用均值填充数值列，用众数填充分类列）
2. 删除明显的异常值（如年龄>150或<0）
3. 确保所有列的数据类型正确
4. 最后将清洗后的DataFrame赋值给变量 df

示例输出格式：
df = df.dropna(subset=['重要列'])
df['数值列'] = df['数值列'].fillna(df['数值列'].mean())

你的代码："""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=prompt)
    
    def clean(self, df: pd.DataFrame, request: str = "自动清洗") -> pd.DataFrame:
        """清洗数据"""
        from data_loader import DataLoader
        data_info = DataLoader.get_data_info(df)
        
        # 生成清洗代码
        result = self.chain.invoke({
            "data_info": data_info,
            "cleaning_request": request
        })
        
        # 解析返回结果
        if isinstance(result, dict):
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
        
        print(f"\n📝 生成的清洗代码:\n{code}")
        
        # 执行代码
        try:
            exec_env = {"pd": pd, "df": df.copy(), "np": __import__('numpy')}
            exec(code, exec_env)
            
            cleaned_df = exec_env.get("df", df)
            
            print(f"✅ 数据清洗完成")
            print(f"清洗前: {df.shape}, 清洗后: {cleaned_df.shape}")
            
            return cleaned_df
        except Exception as e:
            print(f"❌ 清洗失败：{e}")
            print(f"\n代码内容:\n{code}")
            return df

# 测试
if __name__ == "__main__":
    from data_loader import DataLoader
    
    df = DataLoader.load("../data/sales_data.xlsx")
    agent = DataCleaningAgent()
    cleaned_df = agent.clean(df)
    print(f"\n清洗后缺失值:\n{cleaned_df.isnull().sum()}")
