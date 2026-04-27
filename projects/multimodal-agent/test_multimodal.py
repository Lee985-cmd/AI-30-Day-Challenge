"""
测试本地模型的多模态支持
诊断图片识别问题
"""

import os
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


def test_multimodal_support():
    """测试多模态支持"""
    print("=" * 60)
    print("多模态模型诊断测试")
    print("=" * 60)
    
    # 检查环境变量
    local_llm_url = os.getenv("LOCAL_LLM_URL")
    if not local_llm_url:
        print("❌ 未找到 LOCAL_LLM_URL 环境变量")
        return False
    
    print(f"✅ 本地模型地址: {local_llm_url}")
    
    # 使用实际的测试图片
    test_image_path = "test_images/laptop.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ 测试图片不存在: {test_image_path}")
        return False
    
    # 读取并编码图片
    with open(test_image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    print(f"📸 测试图片: {test_image_path}")
    print(f"📏 图片大小: {len(base64_image)} bytes (base64)")
    
    # 初始化模型
    try:
        llm = ChatOpenAI(
            model="qwen-vl-plus",
            openai_api_base=local_llm_url,
            openai_api_key="not-needed",
            temperature=0.3
        )
        print("✅ 模型初始化成功")
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        return False
    
    # 测试1: 标准 OpenAI 格式
    print("\n" + "-" * 60)
    print("测试 1: 标准 OpenAI 多模态格式")
    print("-" * 60)
    
    messages = [
        SystemMessage(content="你是一个图像识别助手。"),
        HumanMessage(content=[
            {"type": "text", "text": "请描述这张图片"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ])
    ]
    
    try:
        response = llm.invoke(messages)
        print(f"✅ 成功！")
        print(f"🤖 回答: {response.content[:200]}...")
        return True
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {str(e)[:200]}")
        
        # 尝试其他格式
        print("\n" + "-" * 60)
        print("测试 2: 简化格式（仅文本）")
        print("-" * 60)
        
        messages_simple = [
            SystemMessage(content="你是一个图像识别助手。"),
            HumanMessage(content="请描述这张图片：笔记本电脑")
        ]
        
        try:
            response = llm.invoke(messages_simple)
            print(f"✅ 简化格式成功！")
            print(f"🤖 回答: {response.content[:200]}...")
            print("\n⚠️  注意: 本地模型可能不支持多模态，只能处理文本")
            return False
        except Exception as e2:
            print(f"❌ 简化格式也失败: {e2}")
            return False


if __name__ == "__main__":
    result = test_multimodal_support()
    
    print("\n" + "=" * 60)
    if result:
        print("✅ 本地模型支持多模态！")
        print("💡 可以正常使用 multimodal_agent.py")
    else:
        print("⚠️  本地模型可能不支持多模态")
        print("\n解决方案：")
        print("1. 确认本地模型是否支持视觉能力（如 qwen-vl, yi-vl 等）")
        print("2. 检查模型文档，确认正确的调用格式")
        print("3. 或使用 OpenAI GPT-4V API")
    print("=" * 60)
