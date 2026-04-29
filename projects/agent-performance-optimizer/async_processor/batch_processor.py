"""
批处理器
合并多个小请求为批量处理，提高效率和降低成本
"""

import asyncio
from typing import List, Callable, Any, Optional
from datetime import datetime
from collections import defaultdict


class BatchProcessor:
    """批处理器 - 自动合并请求进行批量处理"""
    
    def __init__(
        self, 
        batch_size: int = 10,
        wait_time: float = 0.2,
        process_fn: Optional[Callable] = None
    ):
        """
        初始化批处理器
        
        Args:
            batch_size: 批次大小
            wait_time: 等待时间（秒），收集请求的时间窗口
            process_fn: 批量处理函数，接收查询列表，返回结果列表
        """
        self.batch_size = batch_size
        self.wait_time = wait_time
        self.process_fn = process_fn or self._default_process
        self.queue = asyncio.Queue()
        self.running = False
        self.task: Optional[asyncio.Task] = None
        
        # 统计信息
        self.total_batches = 0
        self.total_requests = 0
        self.total_time = 0
    
    async def start(self):
        """启动批处理循环"""
        self.running = True
        self.task = asyncio.create_task(self._batch_loop())
        print(f"✅ 批处理器已启动 (batch_size={self.batch_size}, wait_time={self.wait_time}s)")
    
    async def stop(self):
        """停止批处理器"""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        print("⏹️  批处理器已停止")
    
    async def _batch_loop(self):
        """批处理主循环"""
        while self.running:
            await self._process_batch()
    
    async def _process_batch(self):
        """处理一批请求"""
        batch = []
        futures = []
        
        # 收集请求
        while len(batch) < self.batch_size:
            try:
                future, item = await asyncio.wait_for(
                    self.queue.get(), 
                    timeout=self.wait_time
                )
                batch.append(item)
                futures.append(future)
            except asyncio.TimeoutError:
                break
        
        if not batch:
            return
        
        print(f"📦 处理批次: {len(batch)}个请求")
        
        start_time = datetime.now()
        
        try:
            # 批量处理
            results = await self.process_fn(batch)
            
            # 返回结果
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
            
            # 更新统计
            elapsed = (datetime.now() - start_time).total_seconds()
            self.total_batches += 1
            self.total_requests += len(batch)
            self.total_time += elapsed
        
        except Exception as e:
            # 设置异常
            for future in futures:
                if not future.done():
                    future.set_exception(e)
    
    async def _default_process(self, items: List[Any]) -> List[Any]:
        """
        默认处理函数（需要替换为实际的LLM调用）
        
        Args:
            items: 待处理项列表
            
        Returns:
            处理结果列表
        """
        # 这里应该调用实际的批量LLM API
        # 示例：模拟延迟
        await asyncio.sleep(0.5)
        return [f"处理结果: {item}" for item in items]
    
    async def submit(self, item: Any) -> Any:
        """
        提交请求
        
        Args:
            item: 待处理项
            
        Returns:
            处理结果
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self.queue.put((future, item))
        return await future
    
    async def submit_batch(self, items: List[Any]) -> List[Any]:
        """
        批量提交
        
        Args:
            items: 待处理项列表
            
        Returns:
            处理结果列表
        """
        tasks = [self.submit(item) for item in items]
        return await asyncio.gather(*tasks)
    
    def stats(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        avg_batch_size = (
            self.total_requests / self.total_batches 
            if self.total_batches > 0 
            else 0
        )
        avg_time_per_batch = (
            self.total_time / self.total_batches 
            if self.total_batches > 0 
            else 0
        )
        
        return {
            "total_batches": self.total_batches,
            "total_requests": self.total_requests,
            "avg_batch_size": f"{avg_batch_size:.1f}",
            "avg_time_per_batch": f"{avg_time_per_batch:.2f}s",
            "queue_size": self.queue.qsize()
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_batches = 0
        self.total_requests = 0
        self.total_time = 0


# LLM批处理适配器
class LLMBatchProcessor(BatchProcessor):
    """专门用于LLM调用的批处理器"""
    
    def __init__(
        self,
        api_key: str,
        batch_size: int = 10,
        wait_time: float = 0.2,
        model: str = "gpt-3.5-turbo"
    ):
        super().__init__(batch_size=batch_size, wait_time=wait_time)
        self.api_key = api_key
        self.model = model
    
    async def _default_process(self, queries: List[str]) -> List[str]:
        """批量调用LLM"""
        import aiohttp
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 并发调用（如果API不支持真正的batch）
        tasks = []
        async with aiohttp.ClientSession() as session:
            for query in queries:
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.3
                }
                tasks.append(self._call_single(session, url, headers, data))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常
            final_results = []
            for result in results:
                if isinstance(result, Exception):
                    final_results.append(f"错误: {str(result)}")
                else:
                    final_results.append(result)
            
            return final_results
    
    async def _call_single(self, session, url, headers, data) -> str:
        """单个LLM调用"""
        async with session.post(url, headers=headers, json=data) as response:
            result = await response.json()
            return result['choices'][0]['message']['content']


# 使用示例
async def main():
    # 创建批处理器
    processor = BatchProcessor(batch_size=5, wait_time=0.2)
    
    # 启动批处理
    await processor.start()
    
    # 提交多个请求
    items = [f"任务{i}" for i in range(20)]
    
    start_time = asyncio.get_event_loop().time()
    
    # 并发提交
    tasks = [processor.submit(item) for item in items]
    results = await asyncio.gather(*tasks)
    
    end_time = asyncio.get_event_loop().time()
    
    print(f"\n处理{len(items)}个请求")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"平均每个: {(end_time - start_time) / len(items) * 1000:.0f}ms")
    print(f"\n统计: {processor.stats()}")
    
    # 停止批处理器
    await processor.stop()

if __name__ == "__main__":
    asyncio.run(main())
