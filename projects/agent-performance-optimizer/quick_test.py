"""
快速测试脚本 - 验证项目核心功能
不需要API Key即可运行
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_memory_cache():
    """测试1：内存缓存"""
    print("\n" + "="*80)
    print("测试1：内存缓存 (MemoryCache)")
    print("="*80)
    
    try:
        from cache import MemoryCache
        
        # 创建缓存实例
        cache = MemoryCache(ttl=60, max_size=1000)
        print("✅ 缓存实例创建成功")
        
        # 测试设置和获取
        cache.set("test_key_1", "test_value_1")
        result = cache.get("test_key_1")
        assert result == "test_value_1", f"期望 'test_value_1'，得到 '{result}'"
        print("✅ 基本读写测试通过")
        
        # 测试带上下文的缓存
        cache.set("question", "answer_1", context="context_A")
        cache.set("question", "answer_2", context="context_B")
        
        result_a = cache.get("question", context="context_A")
        result_b = cache.get("question", context="context_B")
        
        assert result_a == "answer_1", "上下文A的缓存错误"
        assert result_b == "answer_2", "上下文B的缓存错误"
        print("✅ 上下文隔离测试通过")
        
        # 测试统计信息
        stats = cache.stats()
        print(f"📊 缓存统计: {stats}")
        assert stats['size'] >= 3, "缓存大小不正确"
        print("✅ 统计功能测试通过")
        
        # 测试删除
        cache.delete("test_key_1")
        assert cache.get("test_key_1") is None, "删除失败"
        print("✅ 删除功能测试通过")
        
        # 测试清空
        cache.clear()
        assert len(cache) == 0, "清空失败"
        print("✅ 清空功能测试通过")
        
        print("\n🎉 内存缓存测试全部通过！\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 内存缓存测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_batch_processor():
    """测试2：批处理器"""
    print("\n" + "="*80)
    print("测试2：批处理器 (BatchProcessor)")
    print("="*80)
    
    try:
        import asyncio
        from async_processor import BatchProcessor
        
        async def run_test():
            # 创建批处理器
            processor = BatchProcessor(batch_size=3, wait_time=0.1)
            
            # 启动批处理
            await processor.start()
            print("✅ 批处理器启动成功")
            
            # 提交测试任务
            tasks = []
            for i in range(5):
                task = processor.submit(f"任务{i+1}")
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks)
            print(f"✅ 处理了 {len(results)} 个任务")
            
            # 检查统计
            stats = processor.stats()
            print(f"📊 批处理统计: {stats}")
            
            # 停止批处理器
            await processor.stop()
            print("✅ 批处理器停止成功")
            
            return True
        
        # 运行异步测试
        result = asyncio.run(run_test())
        
        if result:
            print("\n🎉 批处理器测试全部通过！\n")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"\n❌ 批处理器测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_import_all_modules():
    """测试3：模块导入"""
    print("\n" + "="*80)
    print("测试3：模块导入检查")
    print("="*80)
    
    modules_to_test = [
        ("cache.memory_cache", "MemoryCache"),
        ("cache.redis_cache", "RedisCache"),
        ("cache.semantic_cache", "SemanticCache"),
        ("async_processor.async_agent", "AsyncAgent"),
        ("async_processor.batch_processor", "BatchProcessor"),
    ]
    
    all_passed = True
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ {module_name}.{class_name} 导入成功")
        except ImportError as e:
            print(f"❌ {module_name}.{class_name} 导入失败: {e}")
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有模块导入成功！\n")
    else:
        print("\n⚠️  部分模块导入失败\n")
    
    return all_passed


def test_project_structure():
    """测试4：项目结构"""
    print("\n" + "="*80)
    print("测试4：项目结构检查")
    print("="*80)
    
    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "cache/__init__.py",
        "cache/memory_cache.py",
        "cache/redis_cache.py",
        "cache/semantic_cache.py",
        "async_processor/__init__.py",
        "async_processor/async_agent.py",
        "async_processor/batch_processor.py",
        "benchmark.py",
        "dashboard.py",
        "examples.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} 不存在")
            all_exist = False
    
    if all_exist:
        print("\n🎉 项目结构完整！\n")
    else:
        print("\n⚠️  项目结构不完整\n")
    
    return all_exist


def main():
    """运行所有测试"""
    print("\n" + "🚀"*40)
    print("开始测试 Agent性能优化工具包")
    print("🚀"*40)
    
    results = {}
    
    # 测试1：项目结构
    results['项目结构'] = test_project_structure()
    
    # 测试2：模块导入
    results['模块导入'] = test_import_all_modules()
    
    # 测试3：内存缓存
    results['内存缓存'] = test_memory_cache()
    
    # 测试4：批处理器
    results['批处理器'] = test_batch_processor()
    
    # 打印总结
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s} {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\n总计: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("\n🎉🎉🎉 所有测试通过！项目可以正常使用！🎉🎉🎉\n")
        return 0
    else:
        print(f"\n⚠️  有 {total_tests - passed_tests} 个测试失败，请检查上述错误信息\n")
        return 1


if __name__ == "__main__":
    exit(main())
