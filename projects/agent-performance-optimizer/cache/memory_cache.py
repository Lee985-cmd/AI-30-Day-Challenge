"""
内存缓存实现
适合单机、小规模应用
"""

import hashlib
import time
from typing import Optional, Dict, Any
from threading import Lock


class MemoryCache:
    """基于内存的线程安全缓存"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 10000):
        """
        初始化内存缓存
        
        Args:
            ttl: 缓存过期时间（秒），默认1小时
            max_size: 最大缓存条目数，默认10000
        """
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl
        self.max_size = max_size
        self.lock = Lock()
        
        # 统计信息
        self.hits = 0
        self.misses = 0
    
    def _generate_key(self, query: str, context: str = "") -> str:
        """生成缓存键"""
        content = f"{query}:{context}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, query: str, context: str = "") -> Optional[str]:
        """
        获取缓存
        
        Args:
            query: 查询内容
            context: 上下文信息
            
        Returns:
            缓存的值，如果不存在或已过期则返回None
        """
        key = self._generate_key(query, context)
        
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                
                # 检查是否过期
                if time.time() - timestamp < self.ttl:
                    self.hits += 1
                    return value
                else:
                    # 过期了，删除
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def set(self, query: str, response: str, context: str = "") -> None:
        """
        设置缓存
        
        Args:
            query: 查询内容
            response: 响应内容
            context: 上下文信息
        """
        key = self._generate_key(query, context)
        
        with self.lock:
            # 如果缓存已满，删除最旧的条目
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache, key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
            
            self.cache[key] = (response, time.time())
    
    def delete(self, query: str, context: str = "") -> bool:
        """
        删除缓存
        
        Args:
            query: 查询内容
            context: 上下文信息
            
        Returns:
            是否成功删除
        """
        key = self._generate_key(query, context)
        
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """清空所有缓存"""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def clear_pattern(self, pattern: str) -> int:
        """
        批量删除匹配模式的缓存
        
        Args:
            pattern: 匹配模式（简单实现，只支持前缀匹配）
            
        Returns:
            删除的条目数
        """
        with self.lock:
            keys_to_delete = [
                key for key in self.cache.keys() 
                if key.startswith(pattern)
            ]
            
            for key in keys_to_delete:
                del self.cache[key]
            
            return len(keys_to_delete)
    
    def stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "ttl": self.ttl,
            "usage": f"{len(self.cache) / self.max_size * 100:.2f}%"
        }
    
    def cleanup_expired(self) -> int:
        """
        清理过期的缓存条目
        
        Returns:
            清理的条目数
        """
        current_time = time.time()
        expired_keys = []
        
        with self.lock:
            for key, (_, timestamp) in self.cache.items():
                if current_time - timestamp >= self.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
        
        return len(expired_keys)
    
    def __len__(self) -> int:
        """返回缓存大小"""
        return len(self.cache)
    
    def __contains__(self, query: str) -> bool:
        """检查查询是否在缓存中"""
        return self.get(query) is not None
    
    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"MemoryCache(size={stats['size']}, "
            f"hit_rate={stats['hit_rate']}, "
            f"usage={stats['usage']})"
        )


# 使用示例
if __name__ == "__main__":
    # 创建缓存实例
    cache = MemoryCache(ttl=3600, max_size=1000)
    
    # 设置缓存
    cache.set("什么是AI？", "AI是人工智能...")
    cache.set("Python优点", "Python简单易学...")
    
    # 获取缓存
    result = cache.get("什么是AI？")
    print(f"缓存命中: {result}")
    
    # 查看统计
    print(f"\n缓存统计: {cache.stats()}")
    
    # 测试缓存命中率
    for i in range(100):
        cache.set(f"问题{i}", f"回答{i}")
    
    for i in range(50):
        cache.get(f"问题{i}")  # 这些会命中
    
    for i in range(50, 100):
        cache.get(f"问题{i}")  # 这些也会命中
    
    for i in range(100, 150):
        cache.get(f"问题{i}")  # 这些不会命中
    
    print(f"\n测试后统计: {cache.stats()}")
