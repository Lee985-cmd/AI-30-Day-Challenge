"""
Redis缓存实现
适合分布式、生产环境
"""

import hashlib
import json
import redis
from typing import Optional, Dict, Any, List
from datetime import datetime


class RedisCache:
    """基于Redis的分布式缓存"""
    
    def __init__(
        self, 
        host: str = 'localhost', 
        port: int = 6379, 
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,
        prefix: str = "agent"
    ):
        """
        初始化Redis缓存
        
        Args:
            host: Redis服务器地址
            port: Redis端口
            db: 数据库编号
            password: 密码（可选）
            ttl: 默认过期时间（秒）
            prefix: 键前缀
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        self.ttl = ttl
        self.prefix = prefix
        
        # 测试连接
        try:
            self.redis_client.ping()
            print("✅ Redis连接成功")
        except redis.ConnectionError:
            print("❌ Redis连接失败，请检查配置")
            raise
    
    def _generate_key(self, query: str, context: str = "") -> str:
        """生成缓存键"""
        content = f"{self.prefix}:{query}:{context}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, query: str, context: str = "") -> Optional[str]:
        """
        获取缓存
        
        Args:
            query: 查询内容
            context: 上下文信息
            
        Returns:
            缓存的值，如果不存在则返回None
        """
        key = self._generate_key(query, context)
        value = self.redis_client.get(key)
        
        if value:
            return value
        return None
    
    def set(
        self, 
        query: str, 
        response: str, 
        context: str = "",
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存
        
        Args:
            query: 查询内容
            response: 响应内容
            context: 上下文信息
            ttl: 过期时间（秒），使用默认值如果未指定
            
        Returns:
            是否设置成功
        """
        key = self._generate_key(query, context)
        expire_time = ttl if ttl is not None else self.ttl
        
        try:
            self.redis_client.setex(key, expire_time, response)
            return True
        except Exception as e:
            print(f"❌ 设置缓存失败: {e}")
            return False
    
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
        return self.redis_client.delete(key) > 0
    
    def clear_pattern(self, pattern: str = "*") -> int:
        """
        批量删除匹配模式的缓存
        
        Args:
            pattern: 匹配模式（支持通配符）
            
        Returns:
            删除的条目数
        """
        full_pattern = f"{self.prefix}:{pattern}"
        keys = self.redis_client.keys(full_pattern)
        
        if keys:
            return self.redis_client.delete(*keys)
        return 0
    
    def clear_all(self) -> bool:
        """清空所有当前前缀的缓存"""
        return self.clear_pattern("*") > 0
    
    def exists(self, query: str, context: str = "") -> bool:
        """
        检查缓存是否存在
        
        Args:
            query: 查询内容
            context: 上下文信息
            
        Returns:
            是否存在
        """
        key = self._generate_key(query, context)
        return self.redis_client.exists(key) > 0
    
    def get_ttl(self, query: str, context: str = "") -> int:
        """
        获取缓存剩余生存时间
        
        Args:
            query: 查询内容
            context: 上下文信息
            
        Returns:
            剩余秒数，-1表示永不过期，-2表示不存在
        """
        key = self._generate_key(query, context)
        return self.redis_client.ttl(key)
    
    def stats(self) -> Dict[str, Any]:
        """
        获取Redis服务器统计信息
        
        Returns:
            统计信息字典
        """
        info = self.redis_client.info('stats')
        keyspace = self.redis_client.info('keyspace')
        
        # 计算当前前缀的键数量
        pattern = f"{self.prefix}:*"
        keys = self.redis_client.keys(pattern)
        
        return {
            "type": "Redis",
            "prefix": self.prefix,
            "keys_count": len(keys),
            "total_keys": sum(db.get('keys', 0) for db in keyspace.values()),
            "connected_clients": self.redis_client.info()['connected_clients'],
            "used_memory_human": self.redis_client.info()['used_memory_human'],
            "hit_rate": f"{info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1), 1) * 100:.2f}%"
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        获取内存使用情况
        
        Returns:
            内存使用信息
        """
        info = self.redis_client.info('memory')
        
        return {
            "used_memory": info['used_memory'],
            "used_memory_human": info['used_memory_human'],
            "maxmemory": info['maxmemory'],
            "maxmemory_human": info['maxmemory_human'],
            "mem_fragmentation_ratio": info['mem_fragmentation_ratio']
        }
    
    def ping(self) -> bool:
        """测试连接"""
        try:
            return self.redis_client.ping()
        except:
            return False
    
    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"RedisCache(prefix={stats['prefix']}, "
            f"keys={stats['keys_count']}, "
            f"hit_rate={stats['hit_rate']})"
        )


# 高级功能：JSON序列化缓存
class JSONRedisCache(RedisCache):
    """支持JSON序列化的Redis缓存"""
    
    def get_json(self, query: str, context: str = "") -> Optional[Dict]:
        """获取JSON格式的缓存"""
        value = super().get(query, context)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return None
        return None
    
    def set_json(
        self, 
        query: str, 
        data: Dict, 
        context: str = "",
        ttl: Optional[int] = None
    ) -> bool:
        """设置JSON格式的缓存"""
        try:
            json_str = json.dumps(data, ensure_ascii=False)
            return super().set(query, json_str, context, ttl)
        except Exception as e:
            print(f"❌ JSON序列化失败: {e}")
            return False


# 使用示例
if __name__ == "__main__":
    try:
        # 创建Redis缓存实例
        cache = RedisCache(host='localhost', port=6379, ttl=3600)
        
        # 设置缓存
        cache.set("什么是AI？", "AI是人工智能...")
        cache.set("用户:123:偏好", json.dumps({"language": "Python", "level": "advanced"}))
        
        # 获取缓存
        result = cache.get("什么是AI？")
        print(f"缓存命中: {result}")
        
        # JSON缓存
        json_cache = JSONRedisCache(host='localhost', port=6379)
        json_cache.set_json("用户配置", {"theme": "dark", "lang": "zh"})
        config = json_cache.get_json("用户配置")
        print(f"JSON缓存: {config}")
        
        # 查看统计
        print(f"\n缓存统计: {cache.stats()}")
        print(f"内存使用: {cache.get_memory_usage()}")
        
        # 清理
        cache.clear_all()
        print("\n✅ 缓存已清空")
        
    except redis.ConnectionError:
        print("⚠️  请先启动Redis服务")
        print("   Docker: docker run -d -p 6379:6379 redis")
