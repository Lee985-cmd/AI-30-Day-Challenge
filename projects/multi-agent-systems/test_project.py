"""
multi-agent-systems 项目测试脚本

验证所有模块的基本功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_custom_coordinator():
    """测试1：自定义Agent协调器"""
    print("=" * 60)
    print("测试1: 自定义Agent协调器")
    print("=" * 60)
    
    try:
        from custom_coordinator import AgentCoordinator, Agent
        
        # 创建协调器
        coordinator = AgentCoordinator()
        
        # 创建测试Agent
        def simple_process(task, context):
            return f"处理完成: {task}"
        
        agent = Agent(
            name="test_agent",
            role="测试员",
            expertise="测试",
            process_fn=simple_process
        )
        
        # 添加Agent
        coordinator.add_agent(agent)
        
        # 执行任务
        result = coordinator.assign_task("测试任务", "test_agent")
        
        assert "处理完成" in result, "任务执行结果不正确"
        
        # 获取执行摘要
        summary = coordinator.get_execution_summary()
        assert summary["total_tasks"] == 1, "任务计数错误"
        
        print("✅ 自定义协调器测试通过")
        print(f"   - Agent添加: OK")
        print(f"   - 任务执行: OK")
        print(f"   - 执行摘要: OK")
        return True
        
    except Exception as e:
        print(f"❌ 自定义协调器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_import_dependencies():
    """测试2：依赖导入"""
    print("\n" + "=" * 60)
    print("测试2: 依赖导入检查")
    print("=" * 60)
    
    dependencies = {
        "crewai": "CrewAI框架",
        "langchain": "LangChain",
        "langchain_openai": "LangChain OpenAI",
        "langgraph": "LangGraph"
    }
    
    results = {}
    
    for package, name in dependencies.items():
        try:
            __import__(package)
            print(f"✅ {name} ({package}) - 已安装")
            results[package] = True
        except ImportError as e:
            print(f"⚠️  {name} ({package}) - 未安装")
            results[package] = False
    
    # 至少custom_coordinator不需要外部依赖，应该能通过
    if results.get("crewai", False):
        print("\n✅ 所有依赖已安装，可以运行完整示例")
    else:
        print("\n⚠️  部分依赖未安装，只能运行custom_coordinator示例")
        print("   安装依赖: pip install -r requirements.txt")
    
    return True


def test_file_structure():
    """测试3：文件结构"""
    print("\n" + "=" * 60)
    print("测试3: 项目文件结构")
    print("=" * 60)
    
    required_files = [
        "README.md",
        "requirements.txt",
        "investment_research.py",
        "content_creation.py",
        "custom_coordinator.py"
    ]
    
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file} ({size:,} bytes)")
        else:
            print(f"❌ {file} - 缺失")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ 发现 {len(missing_files)} 个缺失文件")
        return False
    else:
        print("\n✅ 所有必需文件存在")
        return True


def test_code_syntax():
    """测试4：代码语法检查"""
    print("\n" + "=" * 60)
    print("测试4: 代码语法检查")
    print("=" * 60)
    
    python_files = [
        "investment_research.py",
        "content_creation.py",
        "custom_coordinator.py"
    ]
    
    all_valid = True
    
    for file in python_files:
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 编译检查
            compile(code, file_path, 'exec')
            print(f"✅ {file} - 语法正确")
            
        except SyntaxError as e:
            print(f"❌ {file} - 语法错误: {e}")
            all_valid = False
    
    if all_valid:
        print("\n✅ 所有Python文件语法正确")
    else:
        print("\n❌ 存在语法错误")
    
    return all_valid


def test_readme_content():
    """测试5：README内容检查"""
    print("\n" + "=" * 60)
    print("测试5: README内容检查")
    print("=" * 60)
    
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        "项目标题": "# 多Agent协作系统" in content,
        "快速开始": "## 🚀 快速开始" in content,
        "安装说明": "pip install" in content,
        "运行示例": "python" in content,
        "项目结构": "## 📁 项目结构" in content,
        "相关链接": "## 🔗 相关链接" in content
    }
    
    all_passed = True
    for check_name, result in checks.items():
        if result:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_passed = False
    
    if all_passed:
        print("\n✅ README内容完整")
    else:
        print("\n⚠️  README内容不完整")
    
    return all_passed


def main():
    """主测试函数"""
    print("\n🧪 multi-agent-systems 项目测试\n")
    
    results = []
    
    # 运行所有测试
    results.append(("文件结构", test_file_structure()))
    results.append(("代码语法", test_code_syntax()))
    results.append(("README内容", test_readme_content()))
    results.append(("依赖导入", test_import_dependencies()))
    results.append(("自定义协调器", test_custom_coordinator()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！项目状态良好。")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查问题。")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
