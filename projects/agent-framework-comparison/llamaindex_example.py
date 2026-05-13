"""
LlamaIndex 完整示例
展示LlamaIndex的核心功能：数据索引、RAG、查询引擎
"""

import os
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

# 加载环境变量
load_dotenv()

# 配置全局设置
Settings.llm = OpenAI(
    model=os.getenv("MODEL_NAME", "gpt-3.5-turbo"),
    api_key=os.getenv("OPENAI_API_KEY")
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)


# ========== 示例1: 从文本创建索引 ==========
print("=" * 60)
print("示例1: LlamaIndex 基础索引")
print("=" * 60)

# 创建示例文档
sample_texts = [
    """
    Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年首次发布。
    Python的设计哲学强调代码的可读性和简洁的语法，允许程序员用更少的代码表达概念。
    Python支持多种编程范式，包括面向对象、命令式、函数式和过程式编程。
    """,
    
    """
    人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
    这些任务包括视觉感知、语音识别、决策制定和语言翻译等。
    机器学习是AI的一个重要子领域，它使计算机能够从数据中学习而不需要明确编程。
    """,
    
    """
    LangChain是一个用于开发由大型语言模型（LLM）驱动的应用程序的框架。
    它提供了构建Agent、链和工具的组件，使开发者能够轻松创建复杂的LLM应用。
    LangChain支持多种LLM提供商，包括OpenAI、Anthropic和本地模型。
    """,
    
    """
    RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的技术。
    它首先从知识库中检索相关文档，然后使用LLM基于这些文档生成回答。
    RAG可以减少LLM的幻觉问题，并提供基于事实的回答。
    """
]

# 保存示例文本到临时文件
os.makedirs("sample_docs", exist_ok=True)
for i, text in enumerate(sample_texts):
    with open(f"sample_docs/doc_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)

# 加载文档
documents = SimpleDirectoryReader("sample_docs").load_data()
print(f"\n加载了 {len(documents)} 个文档")

# 创建索引
print("正在创建向量索引...")
index = VectorStoreIndex.from_documents(
    documents,
    transformations=[SentenceSplitter(chunk_size=512, chunk_overlap=50)]
)

# 保存索引
index.storage_context.persist(persist_dir="./storage")
print("索引已保存到 ./storage")


# ========== 示例2: 基本查询 ==========
print("\n" + "=" * 60)
print("示例2: LlamaIndex 基本查询")
print("=" * 60)

query_engine = index.as_query_engine(
    similarity_top_k=2,
)

questions = [
    "Python是什么？",
    "什么是人工智能？",
    "LangChain有什么作用？",
    "RAG技术如何解决幻觉问题？"
]

for question in questions:
    print(f"\n问题: {question}")
    response = query_engine.query(question)
    print(f"回答: {response}")
    print(f"相关度分数: {response.metadata.get('score', 'N/A') if hasattr(response, 'metadata') else 'N/A'}")


# ========== 示例3: 高级查询引擎 ==========
print("\n" + "=" * 60)
print("示例3: LlamaIndex 高级查询引擎")
print("=" * 60)

# 创建带后处理器的查询引擎
advanced_query_engine = RetrieverQueryEngine.from_args(
    retriever=VectorIndexRetriever(
        index=index,
        similarity_top_k=3
    ),
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.7)
    ]
)

print("\n使用高级查询引擎:")
question = "Python和AI有什么关系？"
print(f"问题: {question}")
response = advanced_query_engine.query(question)
print(f"回答: {response}")


# ========== 示例4: 流式查询 ==========
print("\n" + "=" * 60)
print("示例4: LlamaIndex 流式查询")
print("=" * 60)

streaming_query_engine = index.as_query_engine(
    similarity_top_k=2,
    streaming=True
)

question = "请详细介绍一下机器学习"
print(f"\n问题: {question}")
print("回答（流式输出）:")

response_stream = streaming_query_engine.query(question)
response_stream.print_response_stream()
print()  # 换行


# ========== 示例5: 自定义提示模板 ==========
print("\n" + "=" * 60)
print("示例5: LlamaIndex 自定义提示模板")
print("=" * 60)

from llama_index.core import PromptTemplate

custom_template = PromptTemplate("""
你是一个专业的技术顾问。基于以下上下文信息回答问题。

上下文信息：
{context_str}

问题：{query_str}

请以简洁、专业的方式回答，如果上下文中没有足够信息，请明确说明。

回答：
""")

custom_query_engine = index.as_query_engine(
    similarity_top_k=2,
    text_qa_template=custom_template
)

question = "我应该学习Python还是直接学习AI？"
print(f"\n问题: {question}")
response = custom_query_engine.query(question)
print(f"回答: {response}")


# ========== 示例6: 索引比较 ==========
print("\n" + "=" * 60)
print("示例6: LlamaIndex 不同索引类型对比")
print("=" * 60)

# 创建列表索引
from llama_index.core import ListIndex

list_index = ListIndex.from_documents(documents)
list_query_engine = list_index.as_query_engine()

print("\n列表索引查询:")
question = "什么是LangChain？"
print(f"问题: {question}")
response = list_query_engine.query(question)
print(f"回答: {response}")

# 对比向量索引
print("\n向量索引查询:")
vector_query_engine = index.as_query_engine()
response = vector_query_engine.query(question)
print(f"回答: {response}")


# ========== 示例7: 元数据过滤 ==========
print("\n" + "=" * 60)
print("示例7: LlamaIndex 元数据过滤")
print("=" * 60)

# 为文档添加元数据
for i, doc in enumerate(documents):
    doc.metadata["source"] = f"doc_{i}"
    doc.metadata["category"] = ["programming", "ai", "framework", "technique"][i]

# 重建索引
index_with_metadata = VectorStoreIndex.from_documents(documents)

# 创建带过滤的查询引擎
filtered_query_engine = index_with_metadata.as_query_engine(
    similarity_top_k=2,
    filters={
        "category": {"$in": ["programming", "ai"]}
    }
)

print("\n使用元数据过滤查询:")
question = "有哪些编程相关的技术？"
print(f"问题: {question}")
response = filtered_query_engine.query(question)
print(f"回答: {response}")


# 清理临时文件
import shutil
if os.path.exists("sample_docs"):
    shutil.rmtree("sample_docs")

print("\n" + "=" * 60)
print("LlamaIndex 示例完成！")
print("=" * 60)
