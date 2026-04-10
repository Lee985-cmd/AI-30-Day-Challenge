"""
RAG 知识库问答系统 - 配置文件
"""

import os
from pathlib import Path

# ========== 基础配置 ==========

# 项目根目录
BASE_DIR = Path(__file__).parent

# 数据目录
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "documents"
CACHE_DIR = BASE_DIR / "cache"

# 确保目录存在
for dir_path in [DATA_DIR, DOCS_DIR, CACHE_DIR]:
    dir_path.mkdir(exist_ok=True)

# ========== API 配置 ==========

# OpenAI API（默认）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

# 本地模型（可选）
USE_LOCAL_MODEL = False
LOCAL_MODEL_URL = "http://localhost:8000"  # ChatGLM/Qwen 服务地址
LOCAL_MODEL_NAME = "chatglm2-6b"

# ========== 文档处理配置 ==========

# 支持的文档格式
SUPPORTED_FORMATS = ['.pdf', '.txt', '.md', '.docx']

# 文本分块策略
CHUNK_SIZE = 500          # 每个文本块的字符数
CHUNK_OVERLAP = 50        # 块之间的重叠字符数

# 分块方法：'recursive' 或 'fixed'
SPLITTER_METHOD = 'recursive'

# ========== 向量数据库配置 ==========

# 向量数据库类型：'chroma' 或 'faiss'
VECTOR_DB_TYPE = 'chroma'

# ChromaDB 配置
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"

# FAISS 配置（如果需要）
FAISS_INDEX_FILE = DATA_DIR / "faiss_index.pkl"

# ========== 检索配置 ==========

# 检索最相关的 K 个片段
TOP_K = 3

# 相似度阈值（0-1），低于此值认为不相关
SIMILARITY_THRESHOLD = 0.7

# 检索策略：'similarity' 或 'mmr'
RETRIEVAL_METHOD = 'similarity'

# MMR 参数（如果使用 mmr 策略）
MMR_LAMBDA = 0.5  # 相关性和多样性的平衡（0-1）

# ========== 问答链配置 ==========

# 生成温度（0-2）
# 低温度（0.1-0.5）：更确定、更可预测
# 中温度（0.6-0.9）：平衡
# 高温度（1.0-2.0）：更有创意但可能不准确
TEMPERATURE = 0.7

# 最大生成 token 数
MAX_TOKENS = 1024

# 系统提示词
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请基于提供的上下文信息回答问题。

规则：
1. 只根据提供的上下文回答，不要编造信息
2. 如果上下文中没有相关信息，请说"根据现有知识库，我无法回答这个问题"
3. 回答要简洁准确，条理清晰
4. 如果涉及代码或命令，请使用代码块格式
"""

# 问答模板
QA_TEMPLATE = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。

上下文：
{context}

问题：{question}

详细回答："""

# ========== 日志配置 ==========

# 日志级别：'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL = 'INFO'

# 日志文件
LOG_FILE = BASE_DIR / "rag_system.log"

# ========== 性能优化 ==========

# 是否启用嵌入缓存
ENABLE_EMBEDDING_CACHE = True

# 缓存目录
EMBEDDING_CACHE_DIR = CACHE_DIR / "embeddings"

# 批量处理大小（避免内存溢出）
BATCH_SIZE = 32

# 是否启用异步处理
ENABLE_ASYNC = False
