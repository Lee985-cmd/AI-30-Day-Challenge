"""
性能基准测试工具
对比优化前后的性能差异
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from cache import MemoryCache
from async_processor import AsyncAgent


class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.results: Dict[str, Any] = {}
    
    def generate_test_queries(self, count: int = 100) -> List[str]:
        """生成测试查询"""
        templates = [
            "什么是{}？",
            "如何学习{}？",
            "{}有哪些优点？",
            "介绍一下{}",
            "{}的未来发展趋势？"
        ]
        
        topics = [
            "AI", "Python", "机器学习", "深度学习", "神经网络",
            "自然语言处理", "计算机视觉", "强化学习", "数据科学",
            "云计算", "区块链", "物联网", "大数据", "网络安全"
        ]
        
        queries = []
        for i in range(count):
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            queries.append(template.format(topic))
        
        return queries
    
    async def benchmark_naive(self, queries: List[str], sample_size: int = 10) -> Dict[str, float]:
        """
        测试原始实现（无优化）
        
        Args:
            queries: 查询列表
            sample_size: 采样数量
            
        Returns:
            性能指标
        """
        print("\n📊 测试原始实现（无优化）...")
        
        times = []
        async with AsyncAgent(api_key=self.api_key) as agent:
            for query in queries[:sample_size]:
                start = time.time()
                try:
                    response = await agent.call_llm_async(query)
                    elapsed = time.time() - start
                    times.append(elapsed)
                    print(f"  ✓ {query[:30]}... ({elapsed:.2f}s)")
                except Exception as e:
                    print(f"  ✗ {query[:30]}... (错误: {e})")
        
        if not times:
            return {"error": "没有成功的请求"}
        
        return {
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "p95_time": sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0],
            "total_requests": len(times),
            "success_rate": len(times) / sample_size * 100
        }
    
    async def benchmark_with_cache(self, queries: List[str], sample_size: int = 20) -> Dict[str, float]:
        """
        测试带缓存的实现
        
        Args:
            queries: 查询列表
            sample_size: 采样数量
            
        Returns:
            性能指标
        """
        print("\n📊 测试带缓存的实现...")
        
        cache = MemoryCache(ttl=3600)
        times = []
        cache_hits = 0
        
        async with AsyncAgent(api_key=self.api_key) as agent:
            # 第一轮：填充缓存
            print("  第一轮：填充缓存...")
            for query in queries[:sample_size//2]:
                start = time.time()
                
                # 检查缓存
                cached = cache.get(query)
                if cached:
                    cache_hits += 1
                    elapsed = time.time() - start
                    times.append(elapsed)
                else:
                    response = await agent.call_llm_async(query)
                    cache.set(query, response)
                    elapsed = time.time() - start
                    times.append(elapsed)
            
            # 第二轮：测试缓存命中
            print("  第二轮：测试缓存命中...")
            for query in queries[:sample_size//2]:
                start = time.time()
                
                cached = cache.get(query)
                if cached:
                    cache_hits += 1
                    elapsed = time.time() - start
                    times.append(elapsed)
                else:
                    response = await agent.call_llm_async(query)
                    cache.set(query, response)
                    elapsed = time.time() - start
                    times.append(elapsed)
        
        hit_rate = cache_hits / len(times) * 100 if times else 0
        
        return {
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "p95_time": sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0],
            "total_requests": len(times),
            "cache_hits": cache_hits,
            "cache_hit_rate": hit_rate
        }
    
    async def benchmark_concurrent(self, queries: List[str], concurrency: int = 10) -> Dict[str, float]:
        """
        测试并发实现
        
        Args:
            queries: 查询列表
            concurrency: 并发数
            
        Returns:
            性能指标
        """
        print(f"\n📊 测试并发实现 (concurrency={concurrency})...")
        
        start_time = time.time()
        
        async with AsyncAgent(api_key=self.api_key) as agent:
            results = await agent.process_multiple_queries(
                queries[:concurrency],
                concurrency=concurrency
            )
        
        total_time = time.time() - start_time
        avg_time = total_time / len(results)
        
        return {
            "total_time": total_time,
            "avg_time_per_request": avg_time,
            "requests_per_second": len(results) / total_time,
            "concurrency": concurrency,
            "total_requests": len(results)
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """
        运行完整基准测试
        
        Returns:
            完整的测试结果
        """
        print("=" * 80)
        print("🚀 Agent性能基准测试")
        print("=" * 80)
        
        # 生成测试数据
        queries = self.generate_test_queries(100)
        print(f"\n✅ 生成了{len(queries)}个测试查询")
        
        # 测试1：原始实现
        naive_results = await self.benchmark_naive(queries, sample_size=5)
        self.results['naive'] = naive_results
        
        # 测试2：带缓存
        cache_results = await self.benchmark_with_cache(queries, sample_size=20)
        self.results['with_cache'] = cache_results
        
        # 测试3：并发
        concurrent_results = await self.benchmark_concurrent(queries, concurrency=10)
        self.results['concurrent'] = concurrent_results
        
        # 打印总结
        self.print_summary()
        
        return self.results
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "=" * 80)
        print("📈 性能测试总结")
        print("=" * 80)
        
        if 'naive' in self.results:
            naive = self.results['naive']
            print(f"\n1️⃣  原始实现:")
            print(f"   平均响应时间: {naive.get('avg_time', 0)*1000:.0f}ms")
            print(f"   P95响应时间: {naive.get('p95_time', 0)*1000:.0f}ms")
            print(f"   成功率: {naive.get('success_rate', 0):.1f}%")
        
        if 'with_cache' in self.results:
            cache = self.results['with_cache']
            print(f"\n2️⃣  带缓存实现:")
            print(f"   平均响应时间: {cache.get('avg_time', 0)*1000:.0f}ms")
            print(f"   缓存命中率: {cache.get('cache_hit_rate', 0):.1f}%")
            
            if 'naive' in self.results:
                improvement = naive['avg_time'] / cache['avg_time'] if cache['avg_time'] > 0 else 0
                print(f"   性能提升: {improvement:.1f}x")
        
        if 'concurrent' in self.results:
            concurrent = self.results['concurrent']
            print(f"\n3️⃣  并发实现:")
            print(f"   总耗时: {concurrent.get('total_time', 0):.2f}s")
            print(f"   QPS: {concurrent.get('requests_per_second', 0):.1f}")
            print(f"   并发数: {concurrent.get('concurrency', 0)}")
        
        print("\n" + "=" * 80)


# 主函数
async def main():
    import os
    
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key')
    
    benchmark = PerformanceBenchmark(api_key=api_key)
    results = await benchmark.run_full_benchmark()
    
    # 保存结果
    import json
    with open('benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n💾 测试结果已保存到 benchmark_results.json")

if __name__ == "__main__":
    asyncio.run(main())
