"""
多模态智能客服 - Streamlit Web 界面
"""

import streamlit as st
from multimodal_agent import MultimodalAgent
from PIL import Image
import os
import tempfile


# 页面配置
st.set_page_config(
    page_title="多模态智能客服",
    page_icon="🤖",
    layout="wide"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: #F5F5F5;
        border-left: 4px solid #4CAF50;
    }
    .uploaded-image {
        max-width: 400px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """初始化会话状态"""
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_image" not in st.session_state:
        st.session_state.current_image = None


def init_agent():
    """初始化 Agent"""
    try:
        st.session_state.agent = MultimodalAgent()
        return True
    except Exception as e:
        st.error(f"❌ Agent 初始化失败: {str(e)}")
        return False


def display_chat_history():
    """显示对话历史"""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.container():
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 用户：</strong>
                    <p>{msg["content"]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if msg.get("image"):
                    st.image(msg["image"], width=300)
        
        elif msg["role"] == "assistant":
            with st.container():
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 助手：</strong>
                    <p>{msg["content"]}</p>
                </div>
                """, unsafe_allow_html=True)


def main():
    """主函数"""
    # 初始化会话状态
    initialize_session_state()
    
    # 标题
    st.markdown('<div class="main-header">🤖 多模态智能客服系统</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">支持图像识别 + 智能对话的客户服务助手</div>', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 功能选择")
        
        mode = st.radio(
            "选择模式：",
            ["💬 智能对话", "🔍 产品识别", "🛠️ 问题诊断"],
            index=0
        )
        
        st.divider()
        
        if st.button("🗑️ 清空对话"):
            st.session_state.messages = []
            st.session_state.current_image = None
            if st.session_state.agent:
                st.session_state.agent.clear_history()
            st.rerun()
        
        st.divider()
        
        st.info("""
        **使用说明：**
        
        1. **智能对话**：上传图片并提问，或纯文字对话
        2. **产品识别**：上传产品图片，自动识别信息
        3. **问题诊断**：上传问题照片，获取解决方案
        
        **支持的图片格式：**
        - JPG/JPEG
        - PNG
        - GIF
        """)
    
    # 初始化 Agent
    if st.session_state.agent is None:
        if not init_agent():
            st.stop()
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 显示对话历史
        display_chat_history()
        
        # 输入区域
        st.divider()
        
        uploaded_file = st.file_uploader(
            "📷 上传图片（可选）",
            type=["jpg", "jpeg", "png", "gif"]
        )
        
        user_input = st.text_area(
            "💬 输入您的问题：",
            placeholder="例如：这个产品有什么问题？如何解决？\n\n💡 提示：由于本地模型不支持图片识别，请详细描述图片内容",
            height=100
        )
        
        col_send, col_clear = st.columns([1, 4])
        
        with col_send:
            send_button = st.button("📤 发送", use_container_width=True, type="primary")
        
        with col_clear:
            if st.button("🔄 清空输入", use_container_width=True):
                st.rerun()
    
    with col2:
        st.header("📸 当前图片")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.current_image = image
            st.image(image, use_container_width=True)
            
            # 保存图片到临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name, format="JPEG")
                temp_image_path = tmp.name
        else:
            st.info("暂无图片")
            temp_image_path = None
    
    # 处理发送
    if send_button and user_input:
        agent = st.session_state.agent
        
        with st.spinner("🤔 AI 正在思考..."):
            try:
                if temp_image_path:
                    # 带图片的对话
                    response = agent.chat_with_image(temp_image_path, user_input)
                    
                    # 添加消息到历史
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_input,
                        "image": st.session_state.current_image
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                else:
                    # 纯文本对话
                    response = agent.chat_text_only(user_input)
                    
                    # 添加消息到历史
                    st.session_state.messages.append({
                        "role": "user",
                        "content": user_input
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 处理失败: {str(e)}")
    
    # 产品识别模式
    if mode == "🔍 产品识别" and temp_image_path:
        with st.spinner("🔍 正在识别产品..."):
            try:
                product_info = agent.get_product_info(temp_image_path)
                
                st.success("✅ 识别完成！")
                
                if "raw_response" in product_info:
                    st.markdown("### 识别结果")
                    st.write(product_info["raw_response"])
                else:
                    st.markdown("### 📦 产品信息")
                    
                    col_info1, col_info2 = st.columns(2)
                    
                    with col_info1:
                        st.markdown(f"**产品名称：** {product_info.get('product_name', '未知')}")
                        st.markdown(f"**品牌：** {product_info.get('brand', '未知')}")
                        st.markdown(f"**颜色：** {product_info.get('color', '未知')}")
                    
                    with col_info2:
                        st.markdown(f"**用途：** {product_info.get('usage', '未知')}")
                        
                        features = product_info.get('features', [])
                        if features:
                            st.markdown("**特征：**")
                            for feature in features:
                                st.markdown(f"- {feature}")
                
            except Exception as e:
                st.error(f"❌ 识别失败: {str(e)}")
    
    # 问题诊断模式
    if mode == "🛠️ 问题诊断" and temp_image_path:
        description = st.text_area("描述您遇到的问题（可选）：", height=100)
        
        if st.button("🔧 开始诊断", type="primary"):
            with st.spinner("🔍 正在诊断问题..."):
                try:
                    result = agent.diagnose_problem(temp_image_path, description)
                    
                    st.success("✅ 诊断完成！")
                    
                    st.markdown("### 📋 诊断报告")
                    st.markdown(result["diagnosis"])
                    
                    # 保存诊断报告
                    if st.button("💾 保存报告"):
                        report_path = f"diagnosis_report_{len(st.session_state.messages)}.md"
                        with open(report_path, "w", encoding="utf-8") as f:
                            f.write(f"# 问题诊断报告\n\n{result['diagnosis']}")
                        st.success(f"报告已保存到: {report_path}")
                
                except Exception as e:
                    st.error(f"❌ 诊断失败: {str(e)}")


if __name__ == "__main__":
    main()
