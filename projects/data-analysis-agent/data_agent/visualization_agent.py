"""
可视化 Agent
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
import os

class VisualizationAgent:
    """可视化 Agent"""
    
    def __init__(self, api_key=None, base_url=None):
        # 使用本地模型
        self.base_url = base_url or os.getenv("LOCAL_LLM_URL", "")
        self.api_key = api_key or os.getenv("LOCAL_LLM_API_KEY", "not-needed")
        
        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            model="qwen-plus",
            temperature=0,
            timeout=30,  # 增加超时时间
            max_retries=2  # 自动重试2次
        )
        
        # 可视化提示词
        prompt = PromptTemplate(
            input_variables=["data_info", "visualization_request"],
            template="""你是一个数据可视化专家。数据已经加载到变量 df 中（pandas DataFrame）。

数据信息：
{data_info}

可视化需求：
{visualization_request}

请只输出 Python 代码，不要有任何解释、注释或markdown标记。代码要求：
1. 直接使用变量 df，不要使用 pd.read_csv() 读取文件
2. 使用 matplotlib 或 seaborn 绘图
3. 设置中文字体：plt.rcParams['font.sans-serif'] = ['SimHei']
4. 选择合适的图表类型（柱状图/折线图/饼图等）
5. 添加标题和轴标签
6. 保存图表到 './charts/chart.png'，dpi=100, bbox_inches='tight'
7. 图片大小设置为 plt.figure(figsize=(12, 8))

示例输出格式：
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(12, 8))
region_sales = df.groupby('地区')['销售额'].sum()
plt.bar(region_sales.index, region_sales.values)
plt.title('各地区销售额对比')
plt.xlabel('地区')
plt.ylabel('销售额')
plt.savefig('./charts/chart.png', dpi=100, bbox_inches='tight')

你的代码："""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=prompt)
    
    def visualize(self, df: pd.DataFrame, request: str = "自动选择合适的图表") -> str:
        """生成可视化图表"""
        from data_loader import DataLoader
        data_info = DataLoader.get_data_info(df)
        
        # 生成可视化代码
        result = self.chain.invoke({
            "data_info": data_info,
            "visualization_request": request
        })
        
        code = result["text"] if isinstance(result, dict) else str(result)
        
        # 执行代码
        try:
            os.makedirs("./charts", exist_ok=True)
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            exec_env = {
                "pd": pd, 
                "plt": plt, 
                "sns": sns, 
                "df": df,
                "np": __import__('numpy')
            }
            exec(code, exec_env)
            
            chart_path = "./charts/chart.png"
            print(f"✅ 图表已保存到: {chart_path}")
            
            return chart_path
        except Exception as e:
            print(f"❌ 可视化失败：{e}")
            return None

# 测试
if __name__ == "__main__":
    from data_loader import DataLoader
    
    df = DataLoader.load("../data/sales_data.xlsx")
    agent = VisualizationAgent()
    chart_path = agent.visualize(df, "绘制各地区销售额对比柱状图")
