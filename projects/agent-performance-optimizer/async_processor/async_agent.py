"""
异步Agent实现
使用asyncio实现并发LLM调用
"""

import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from datetime import datetime


class AsyncAgent:
    """异步Agent，支持并发LLM调用"""
    
    def __init__(
        self, 
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-3.5-turbo",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        初始化异步Agent
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_time = 0
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def call_llm_async(
        self, 
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """
        异步调用LLM
        
        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            LLM响应文本
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        start_time = datetime.now()
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        content = result['choices'][0]['message']['content']
                        
                        # 更新统计
                        elapsed = (datetime.now() - start_time).total_seconds()
                        self.total_requests += 1
                        self.successful_requests += 1
                        self.total_time += elapsed
                        
                        return content
                    else:
                        error_text = await response.text()
                        raise Exception(f"API错误: {response.status} - {error_text}")
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.total_requests += 1
                    self.failed_requests += 1
                    raise Exception(f"调用失败（已重试{self.max_retries}次）: {e}")
                
                # 指数退避等待
                wait_time = 2 ** attempt
                print(f"⚠️  第{attempt + 1}次重试，等待{wait_time}秒...")
                await asyncio.sleep(wait_time)
    
    async def process_multiple_queries(
        self, 
        queries: List[str],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        concurrency: int = 10
    ) -> List[str]:
        """
        并行处理多个查询
        
        Args:
            queries: 查询列表
            temperature: 温度参数
            max_tokens: 最大token数
            concurrency: 最大并发数
            
        Returns:
            响应列表
        """
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_call(query: str) -> str:
            async with semaphore:
                try:
                    return await self.call_llm_async(query, temperature, max_tokens)
                except Exception as e:
                    return f"错误: {str(e)}"
        
        tasks = [limited_call(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(f"查询{i}失败: {str(result)}")
            else:
                final_results.append(result)
        
        return final_results
    
    async def stream_llm_response(
        self, 
        prompt: str,
        callback=None
    ) -> str:
        """
        流式获取LLM响应
        
        Args:
            prompt: 提示词
            callback: 每收到一个token时的回调函数
            
        Returns:
            完整响应文本
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        full_text = ""
        
        async with self.session.post(url, headers=headers, json=data) as response:
            async for line in response.content:
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            import json
                            data = json.loads(data_str)
                            delta = data['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            full_text += content
                            
                            # 调用回调
                            if callback:
                                callback(content)
                        
                        except json.JSONDecodeError:
                            continue
        
        return full_text
    
    def stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        avg_time = (
            self.total_time / self.successful_requests 
            if self.successful_requests > 0 
            else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": f"{self.successful_requests / max(self.total_requests, 1) * 100:.2f}%",
            "avg_response_time": f"{avg_time:.2f}s",
            "total_time": f"{self.total_time:.2f}s"
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_time = 0


# 使用示例
async def main():
    import os
    
    api_key = os.getenv('OPENAI_API_KEY', 'your-api-key')
    
    async with AsyncAgent(api_key=api_key) as agent:
        # 单个查询
        print("🔍 测试单个查询...")
        response = await agent.call_llm_async("什么是AI？")
        print(f"回答: {response[:100]}...\n")
        
        # 多个查询（并发）
        print("🔍 测试并发查询...")
        queries = [
            "Python有哪些优点？",
            "如何学习编程？",
            "AI的未来发展趋势？",
            "推荐一本好书",
            "什么是机器学习？"
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await agent.process_multiple_queries(queries, concurrency=5)
        end_time = asyncio.get_event_loop().time()
        
        print(f"处理{len(queries)}个查询")
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"平均每个: {(end_time - start_time) / len(queries) * 1000:.0f}ms\n")
        
        for i, (query, result) in enumerate(zip(queries, results), 1):
            print(f"{i}. {query}")
            print(f"   → {result[:50]}...\n")
        
        # 查看统计
        print(f"📊 统计信息: {agent.stats()}")

if __name__ == "__main__":
    asyncio.run(main())
