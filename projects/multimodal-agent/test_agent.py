"""
多模态 Agent 测试脚本
"""

from multimodal_agent import MultimodalAgent
import os


def test_init():
    """测试初始化"""
    print("测试 1: Agent 初始化")
    print("-" * 50)
    
    try:
        agent = MultimodalAgent()
        print("✅ Agent 初始化成功")
        return agent
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return None


def test_text_chat(agent):
    """测试纯文本对话"""
    print("\n测试 2: 纯文本对话")
    print("-" * 50)
    
    question = "你好，请介绍一下你自己"
    print(f"用户: {question}")
    
    try:
        response = agent.chat_text_only(question)
        print(f"AI: {response[:200]}...")
        print("✅ 文本对话成功")
    except Exception as e:
        print(f"❌ 对话失败: {e}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("多模态智能客服 Agent - 测试")
    print("=" * 60)
    
    # 测试初始化
    agent = test_init()
    
    if not agent:
        print("\n⚠️  请先配置 LOCAL_LLM_URL 或 OPENAI_API_KEY")
        return
    
    # 测试文本对话
    test_text_chat(agent)
    
    print("\n" + "=" * 60)
    print("✅ 基础测试完成！")
    print("\n💡 下一步：")
    print("1. 准备测试图片")
    print("2. 运行 streamlit_app.py 启动Web界面")
    print("3. 上传图片和进行对话测试")
    print("=" * 60)


if __name__ == "__main__":
    main()
