"""
测试数据分析 Agent 功能
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_agent'))

from data_loader import DataLoader
from pandas_agent import PandasAIAgent
from cleaning_agent import DataCleaningAgent
from visualization_agent import VisualizationAgent

def test_data_loading():
    """测试数据加载"""
    print("=" * 60)
    print("测试 1: 数据加载")
    print("=" * 60)
    
    df = DataLoader.load("data/sales_data.xlsx")
    print(f"✅ 数据加载成功: {df.shape}")
    print(f"\n前 5 行:\n{df.head()}")
    return df

def test_pandas_agent(df):
    """测试 Pandas Agent"""
    print("\n" + "=" * 60)
    print("测试 2: Pandas Agent 查询")
    print("=" * 60)
    
    agent = PandasAIAgent()
    
    questions = [
        "哪个地区的销售额最高？",
        "各品类的平均客单价是多少？"
    ]
    
    for q in questions:
        print(f"\n问题: {q}")
        result = agent.query(df, q)
        if result["success"]:
            print(f"✅ 回答: {result['result'][:200]}")
        else:
            print(f"❌ 失败: {result['error']}")

def test_cleaning_agent(df):
    """测试数据清洗"""
    print("\n" + "=" * 60)
    print("测试 3: 数据清洗")
    print("=" * 60)
    
    agent = DataCleaningAgent()
    cleaned_df = agent.clean(df)
    print(f"✅ 清洗完成: {cleaned_df.shape}")
    print(f"缺失值统计:\n{cleaned_df.isnull().sum()}")

def test_visualization_agent(df):
    """测试可视化"""
    print("\n" + "=" * 60)
    print("测试 4: 数据可视化")
    print("=" * 60)
    
    agent = VisualizationAgent()
    chart_path = agent.visualize(df, "绘制各地区销售额对比柱状图")
    
    if chart_path and os.path.exists(chart_path):
        print(f"✅ 图表已保存: {chart_path}")
    else:
        print("❌ 图表生成失败")

if __name__ == "__main__":
    # 检查 API Key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 错误: 未设置 DASHSCOPE_API_KEY 环境变量")
        print("\n请运行以下命令设置 API Key:")
        print('[System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-your-api-key", "User")')
        sys.exit(1)
    
    try:
        # 测试数据加载
        df = test_data_loading()
        
        # 测试 Pandas Agent
        test_pandas_agent(df)
        
        # 测试数据清洗
        test_cleaning_agent(df)
        
        # 测试可视化
        test_visualization_agent(df)
        
        print("\n" + "=" * 60)
        print("✅ 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
