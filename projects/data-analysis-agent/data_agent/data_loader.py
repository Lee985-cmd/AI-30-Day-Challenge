"""
数据加载模块
支持 Excel、CSV、SQL 等多种数据源
"""
import pandas as pd
from typing import Union
import os

class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        """
        加载数据文件
        
        Args:
            file_path: 文件路径（支持 .csv, .xlsx, .xls）
            
        Returns:
            DataFrame
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
    
    @staticmethod
    def load_from_sql(query: str, connection_string: str) -> pd.DataFrame:
        """
        从数据库加载数据
        
        Args:
            query: SQL 查询语句
            connection_string: 数据库连接字符串
            
        Returns:
            DataFrame
        """
        try:
            import sqlalchemy
            engine = sqlalchemy.create_engine(connection_string)
            return pd.read_sql(query, engine)
        except ImportError:
            raise ImportError("请安装 sqlalchemy: pip install sqlalchemy")
    
    @staticmethod
    def get_data_info(df: pd.DataFrame) -> str:
        """
        获取数据摘要信息
        
        Args:
            df: 数据框
            
        Returns:
            数据摘要字符串
        """
        info = f"""
数据形状: {df.shape}
列名: {df.columns.tolist()}

数据类型:
{df.dtypes.to_string()}

缺失值统计:
{df.isnull().sum().to_string()}

前 5 行数据:
{df.head().to_string()}

数值列统计:
{df.describe().to_string()}
        """
        return info

# 测试
if __name__ == "__main__":
    # 生成示例数据
    from generate_sample_data import generate_sales_data
    
    print("生成示例数据...")
    df = generate_sales_data()
    
    print("\n数据信息:")
    print(DataLoader.get_data_info(df))
    
    # 保存到 Excel
    os.makedirs("data", exist_ok=True)
    df.to_excel("data/sales_data.xlsx", index=False)
    print("\n✅ 示例数据已保存到 data/sales_data.xlsx")
