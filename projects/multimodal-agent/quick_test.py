"""
快速测试多模态 Agent
使用生成的测试图片进行自动化测试
"""

from multimodal_agent import MultimodalAgent
import os


def test_product_recognition(agent):
    """测试产品识别"""
    print("\n" + "=" * 60)
    print("测试 1: 产品识别")
    print("=" * 60)
    
    test_cases = [
        ("test_images/laptop.jpg", "请识别这个产品"),
        ("test_images/phone.jpg", "这是什么设备？"),
    ]
    
    for image_path, question in test_cases:
        if not os.path.exists(image_path):
            print(f"⚠️  跳过: {image_path} 不存在")
            continue
        
        print(f"\n📸 图片: {image_path}")
        print(f"❓ 问题: {question}")
        
        try:
            response = agent.analyze_image(image_path, question)
            print(f"🤖 回答: {response[:200]}...")
            print("✅ 成功")
        except Exception as e:
            print(f"❌ 失败: {e}")


def test_problem_diagnosis(agent):
    """测试问题诊断"""
    print("\n" + "=" * 60)
    print("测试 2: 问题诊断")
    print("=" * 60)
    
    test_cases = [
        ("test_images/screen_crack.jpg", "屏幕裂了怎么办？"),
        ("test_images/battery_issue.jpg", "电池不耐用怎么处理？"),
    ]
    
    for image_path, question in test_cases:
        if not os.path.exists(image_path):
            print(f"⚠️  跳过: {image_path} 不存在")
            continue
        
        print(f"\n📸 图片: {image_path}")
        print(f"❓ 问题: {question}")
        
        try:
            result = agent.diagnose_problem(image_path, question)
            print(f"🤖 诊断:\n{result['diagnosis'][:300]}...")
            print("✅ 成功")
        except Exception as e:
            print(f"❌ 失败: {e}")


def test_conversation(agent):
    """测试对话能力"""
    print("\n" + "=" * 60)
    print("测试 3: 智能对话")
    print("=" * 60)
    
    # 第一轮
    image_path = "test_images/laptop.jpg"
    if os.path.exists(image_path):
        print(f"\n📸 上传图片: {image_path}")
        question1 = "这是什么产品？"
        print(f"❓ 问题 1: {question1}")
        
        try:
            response1 = agent.chat_with_image(image_path, question1)
            print(f"🤖 回答 1: {response1[:150]}...")
            print("✅ 第一轮成功")
        except Exception as e:
            print(f"❌ 第一轮失败: {e}")
            return
        
        # 第二轮（无图片，测试记忆）
        question2 = "它有什么特点？"
        print(f"\n❓ 问题 2: {question2}")
        
        try:
            response2 = agent.chat_text_only(question2)
            print(f"🤖 回答 2: {response2[:150]}...")
            print("✅ 第二轮成功（有记忆）")
        except Exception as e:
            print(f"❌ 第二轮失败: {e}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("多模态 Agent 自动化测试")
    print("=" * 60)
    
    # 检查测试图片
    if not os.path.exists("test_images"):
        print("\n⚠️  测试图片目录不存在")
        print("💡 请先运行: python generate_test_images.py")
        return
    
    # 初始化 Agent
    print("\n🔧 初始化 Agent...")
    try:
        agent = MultimodalAgent()
        print("✅ Agent 初始化成功")
    except Exception as e:
        print(f"❌ Agent 初始化失败: {e}")
        print("\n⚠️  请配置 LOCAL_LLM_URL 或 OPENAI_API_KEY")
        return
    
    # 执行测试
    test_product_recognition(agent)
    test_problem_diagnosis(agent)
    test_conversation(agent)
    
    # 总结
    print("\n" + "=" * 60)
    print("✅ 自动化测试完成！")
    print("=" * 60)
    print("\n💡 下一步：")
    print("1. 查看 TEST_GUIDE.md 了解详细测试方法")
    print("2. 运行 streamlit_app.py 体验 Web 界面")
    print("3. 上传真实照片进行更多测试")
    print("=" * 60)


if __name__ == "__main__":
    main()
