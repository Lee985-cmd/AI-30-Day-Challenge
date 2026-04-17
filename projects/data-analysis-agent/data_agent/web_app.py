"""
Streamlit Web 界面
"""
import streamlit as st
import requests
import pandas as pd
from PIL import Image
import os

st.set_page_config(
    page_title="AI 数据分析 Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI 数据分析 Agent")
st.markdown("---")

# 侧边栏
st.sidebar.title("🎯 功能说明")
st.sidebar.markdown("""
- **上传数据**：支持 Excel、CSV 格式
- **自然语言查询**：用中文提问即可
- **自动清洗**：一键处理缺失值和异常值
- **智能可视化**：自动生成合适的图表
""")

st.sidebar.title("📈 系统状态")
try:
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        st.sidebar.success("✅ 服务运行中")
    else:
        st.sidebar.error("❌ 服务异常")
except:
    st.sidebar.error("❌ 服务未启动，请先运行: python data_agent/api.py")

# 初始化会话状态
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

# 上传文件
st.header("1️⃣ 上传数据文件")
uploaded_file = st.file_uploader("选择 Excel 或 CSV 文件", type=['xlsx', 'xls', 'csv'])

if uploaded_file is not None:
    # 保存文件
    file_path = f"./uploads/{uploaded_file.name}"
    os.makedirs("./uploads", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 上传到服务器
    files = {'file': (uploaded_file.name, uploaded_file.getvalue())}
    response = requests.post("http://localhost:8000/upload", files=files)
    
    if response.status_code == 200:
        result = response.json()
        st.session_state.uploaded = True
        st.success(f"✅ 文件上传成功！数据形状: {result['shape']}")
        
        # 显示数据预览
        st.subheader("数据预览")
        df_preview = pd.DataFrame(result['preview'])
        st.dataframe(df_preview)
    else:
        st.error("❌ 上传失败")

# 数据分析
if st.session_state.uploaded:
    st.markdown("---")
    st.header("2️⃣ 自然语言查询")
    
    question = st.text_input("请输入你的问题（例如：哪个地区销售额最高？）")
    
    if st.button("🔍 查询"):
        if question:
            with st.spinner("正在分析..."):
                response = requests.post(
                    "http://localhost:8000/query",
                    json={"question": question}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.subheader("分析结果")
                    st.write(result.get('result', '无结果'))
                else:
                    st.error("查询失败")
    
    st.markdown("---")
    st.header("3️⃣ 数据清洗")
    
    if st.button("🧹 自动清洗数据"):
        with st.spinner("正在清洗..."):
            response = requests.post(
                "http://localhost:8000/clean",
                json={"request": "自动清洗"}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ 清洗完成！数据形状: {result['shape']}")
            else:
                st.error("清洗失败")
    
    st.markdown("---")
    st.header("4️⃣ 数据可视化")
    
    viz_request = st.text_input("可视化需求（例如：绘制各地区销售额对比柱状图）", 
                                 value="自动选择合适的图表")
    
    if st.button("📊 生成图表"):
        with st.spinner("正在生成图表..."):
            response = requests.post(
                "http://localhost:8000/visualize",
                json={"request": viz_request}
            )
            
            if response.status_code == 200:
                result = response.json()
                chart_path = result.get('chart_path')
                
                if chart_path and os.path.exists(chart_path):
                    st.subheader("生成的图表")
                    image = Image.open(chart_path)
                    st.image(image, use_column_width=True)
                else:
                    st.error("图表文件不存在")
            else:
                st.error("图表生成失败")

# 使用说明
st.markdown("---")
st.header("💡 使用示例")

st.markdown("""
**常见查询问题：**
- 哪个地区的销售额最高？
- Q3 各品类的销售额对比
- 华东区 Q2 和 Q3 的销售额变化趋势
- 客单价最高的品类是什么？

**可视化需求：**
- 绘制各地区销售额对比柱状图
- 绘制月度销售额趋势折线图
- 绘制各品类销售额占比饼图
""")
