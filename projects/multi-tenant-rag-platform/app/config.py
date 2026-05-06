"""
配置管理
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用信息
    APP_NAME: str = "Multi-Tenant RAG Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # 数据库配置
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/rag_db"
    
    # JWT配置
    JWT_SECRET_KEY: str = "your-secret-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # OpenAI配置
    OPENAI_API_KEY: Optional[str] = None
    
    # 配额默认值
    DEFAULT_MAX_DOCUMENTS: int = 10000
    DEFAULT_MAX_QUERIES_PER_DAY: int = 10000
    DEFAULT_MAX_STORAGE_GB: float = 10.0
    DEFAULT_MAX_CONCURRENT_REQUESTS: int = 10
    
    # 隔离策略阈值
    ROW_ISOLATION_THRESHOLD: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
