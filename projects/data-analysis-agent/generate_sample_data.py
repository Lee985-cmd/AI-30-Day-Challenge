"""
生成示例销售数据
用于演示数据分析 Agent
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data():
    """生成电商销售数据"""
    np.random.seed(42)
    
    # 基础配置
    regions = ['华东', '华南', '华北', '西南', '西北']
    categories = ['电子产品', '服装', '食品', '家居', '图书']
    months = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06',
              '2024-07', '2024-08', '2024-09', '2024-10', '2024-11', '2024-12']
    
    data = []
    
    for month in months:
        for region in regions:
            for category in categories:
                # 生成销售数据（带一些趋势和季节性）
                base_sales = np.random.randint(5000, 20000)
                
                # 添加季节性因素
                month_num = int(month.split('-')[1])
                if month_num in [6, 11, 12]:  # 促销月份
                    base_sales *= 1.3
                elif month_num in [2]:  # 春节影响
                    base_sales *= 0.8
                
                # 添加地区差异
                if region == '华东':
                    base_sales *= 1.2
                elif region == '西北':
                    base_sales *= 0.8
                
                # 添加品类差异
                if category == '电子产品':
                    base_sales *= 1.5
                elif category == '图书':
                    base_sales *= 0.6
                
                sales = int(base_sales)
                orders = int(sales / np.random.randint(80, 150))
                customers = int(orders * np.random.uniform(0.7, 0.9))
                
                data.append({
                    '月份': month,
                    '地区': region,
                    '品类': category,
                    '销售额': sales,
                    '订单数': orders,
                    '客户数': customers,
                    '客单价': round(sales / max(orders, 1), 2)
                })
    
    df = pd.DataFrame(data)
    
    # 添加一些缺失值（模拟真实数据）
    mask = np.random.random(df.shape[0]) < 0.02
    df.loc[mask, '销售额'] = np.nan
    
    mask = np.random.random(df.shape[0]) < 0.01
    df.loc[mask, '客户数'] = np.nan
    
    return df

def generate_customer_data():
    """生成客户数据"""
    np.random.seed(43)
    
    n_customers = 1000
    
    data = {
        '客户ID': range(1, n_customers + 1),
        '姓名': [f'客户{i}' for i in range(1, n_customers + 1)],
        '年龄': np.random.randint(18, 65, n_customers),
        '性别': np.random.choice(['男', '女'], n_customers),
        '地区': np.random.choice(['华东', '华南', '华北', '西南', '西北'], n_customers),
        '注册时间': [
            (datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
            for _ in range(n_customers)
        ],
        '累计消费': np.random.randint(100, 50000, n_customers),
        '订单次数': np.random.randint(1, 50, n_customers),
        '会员等级': np.random.choice(['普通', '银卡', '金卡', '钻石'], n_customers, p=[0.5, 0.3, 0.15, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # 添加一些异常值
    df.loc[10, '年龄'] = 200  # 异常年龄
    df.loc[20, '累计消费'] = -1000  # 异常消费
    
    return df

if __name__ == "__main__":
    print("正在生成销售数据...")
    sales_df = generate_sales_data()
    sales_df.to_excel("data/sales_data.xlsx", index=False)
    print(f"✅ 销售数据已保存: {sales_df.shape}")
    print(sales_df.head())
    
    print("\n正在生成客户数据...")
    customer_df = generate_customer_data()
    customer_df.to_excel("data/customer_data.xlsx", index=False)
    print(f"✅ 客户数据已保存: {customer_df.shape}")
    print(customer_df.head())
    
    print("\n数据统计:")
    print(f"销售数据: {len(sales_df)} 条记录")
    print(f"客户数据: {len(customer_df)} 条记录")
