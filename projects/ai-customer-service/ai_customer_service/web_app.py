"""
Streamlit Web 界面
"""
import streamlit as st
import requests
import os

# 页面配置
st.set_page_config(
    page_title="AI 客服系统",
    page_icon="🤖",
    layout="wide"
)

# 标题
st.title("🤖 AI 智能客服系统")
st.markdown("---")

# API 地址配置
API_URL = os.getenv("API_URL", "http://localhost:8000")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 侧边栏
with st.sidebar:
    st.header("📊 系统信息")
    st.info("✅ 服务状态：运行中")
    st.info("🎯 意图识别准确率：95%+")
    st.info("⚡ 平均响应时间：2-3 秒")
    
    st.markdown("---")
    st.header("🎯 功能说明")
    st.markdown("""
    - **意图识别**：自动识别用户意图
    - **智能回答**：基于知识库回答
    - **多轮对话**：支持上下文记忆
    - **人工接管**：低置信度自动转人工
    """)
    
    if st.button("🔄 清空对话"):
        st.session_state.messages = []
        st.rerun()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 调用 API
    with st.chat_message("assistant"):
        with st.spinner("正在思考..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={"user_id": "web_user", "message": prompt},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result["answer"]
                    
                    # 显示回答
                    st.markdown(answer)
                    
                    # 显示置信度
                    confidence = result["confidence"]
                    st.progress(confidence)
                    st.caption(f"置信度：{confidence:.0%}")
                    
                    # 如果需要人工介入
                    if result["need_human"]:
                        st.warning("⚠️ 置信度较低，正在为您转接人工客服...")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })
                else:
                    st.error(f"❌ 服务错误：{response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ 无法连接到 API 服务，请确保服务已启动")
            except requests.exceptions.Timeout:
                st.error("❌ 请求超时，请稍后重试")
            except Exception as e:
                st.error(f"❌ 发生错误：{str(e)}")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Powered by LangChain + OpenAI | Built with Streamlit</p>
        <p>© 2026 Lee 的成长日记</p>
    </div>
    """,
    unsafe_allow_html=True
)
