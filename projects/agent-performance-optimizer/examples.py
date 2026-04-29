"""
完整使用示例
演示如何使用性能优化工具包
"""

import asyncio
import os
from cache import MemoryCache, RedisCache, SemanticCache
from async_processor import AsyncAgent, BatchProcessor


async def example_1_memory_cache():
    """示例1：内存缓存"""
    print("=" * 80)
    print("示例1：内存缓存")
    print("=" * 80)
    
    # 创建缓存
    cache = MemoryCache(ttl=3600, max_size=10000)
    
    # 设置缓存
    cache.set("问题1", "回答1")
    cache.set("问题2", "回答2")
    
    # 获取缓存
    result = cache.get("问题1")
    print(f"缓存命中: {result}")
    
    # 查看统计
    print(f"缓存统计: {cache.stats()}")
    print()


async def example_2_async_agent():
    """示例2：异步Agent"""
    print("=" * 80)
    print("示例2：异步Agent并发调用")
    print("=" * 80)
    
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key')
    
    queries = [
        "什么是AI？",
        "Python有哪些优点？",
        "如何学习编程？",
    ]
    
    async with AsyncAgent(api_key=api_key) as agent:
        start_time = asyncio.get_event_loop().time()
        
        results = await agent.process_multiple_queries(queries, concurrency=3)
        
        end_time = asyncio.get_event_loop().time()
        
        print(f"处理{len(queries)}个查询")
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"平均每个: {(end_time - start_time) / len(queries) * 1000:.0f}ms")
        
        for query, result in zip(queries, results):
            print(f"\nQ: {query}")
            print(f"A: {result[:100]}...")
    
    print()


async def example_3_batch_processor():
    """示例3：批处理器"""
    print("=" * 80)
    print("示例3：批处理器")
    print("=" * 80)
    
    # 创建批处理器
    processor = BatchProcessor(batch_size=5, wait_time=0.2)
    
    # 启动
    await processor.start()
    
    # 提交请求
    items = [f"任务{i}" for i in range(10)]
    
    start_time = asyncio.get_event_loop().time()
    
    tasks = [processor.submit(item) for item in items]
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    
    print(f"处理{len(items)}个请求")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"统计: {processor.stats()}")
    
    # 停止
    await processor.stop()
    print()


async def example_4_complete_workflow():
    """示例4：完整工作流程（缓存 + 异步）"""
    print("=" * 80)
    print("示例4：完整工作流程")
    print("=" * 80)
    
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key')
    
    # 创建缓存
    cache = MemoryCache(ttl=3600)
    
    # 创建Agent
    async with AsyncAgent(api_key=api_key) as agent:
        queries = [
            "什么是机器学习？",
            "Python的优势？",
            "什么是机器学习？",  # 重复，应该命中缓存
        ]
        
        for query in queries:
            # 检查缓存
            cached = cache.get(query)
            
            if cached:
                print(f"✅ 缓存命中: {query}")
                response = cached
            else:
                print(f"❌ 缓存未命中: {query}")
                response = await agent.call_llm_async(query)
                cache.set(query, response)
            
            print(f"   回答: {response[:50]}...\n")
    
    print(f"最终缓存统计: {cache.stats()}")
    print()


async def main():
    """运行所有示例"""
    print("\n🚀 Agent性能优化工具包 - 使用示例\n")
    
    # 示例1：内存缓存
    await example_1_memory_cache()
    
    # 示例2：异步Agent
    # await example_2_async_agent()  # 需要API Key
    
    # 示例3：批处理器
    await example_3_batch_processor()
    
    # 示例4：完整工作流程
    # await example_4_complete_workflow()  # 需要API Key
    
    print("✅ 所有示例完成！")


if __name__ == "__main__":
    asyncio.run(main())
