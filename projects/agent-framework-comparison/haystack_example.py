"""
Haystack 完整示例
展示Haystack的核心功能：Pipeline、DocumentStore、Retriever、Reader
"""

import os
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import OpenAIGenerator
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.utils import Secret

# 加载环境变量
load_dotenv()

# 配置OpenAI API
api_key = os.getenv("OPENAI_API_KEY", "")


# ========== 示例1: 基础文档存储和检索 ==========
print("=" * 60)
print("示例1: Haystack 基础文档存储")
print("=" * 60)

# 创建文档存储
document_store = InMemoryDocumentStore()

# 创建示例文档
documents = [
    Document(content="Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年首次发布。"),
    Document(content="人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"),
    Document(content="LangChain是一个用于开发由大型语言模型（LLM）驱动的应用程序的框架。"),
    Document(content="RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的技术。"),
    Document(content="机器学习是AI的一个重要子领域，它使计算机能够从数据中学习而不需要明确编程。"),
]

# 写入文档
document_store.write_documents(documents)
print(f"\n已写入 {len(documents)} 个文档到存储")

# 创建BM25检索器
retriever = InMemoryBM25Retriever(document_store=document_store)

# 测试检索
queries = ["Python", "人工智能", "LangChain"]

for query in queries:
    print(f"\n查询: {query}")
    results = retriever.run(query=query, top_k=2)
    
    for i, doc in enumerate(results['documents'], 1):
        print(f"  {i}. {doc.content[:80]}...")


# ========== 示例2: 向量检索 ==========
print("\n" + "=" * 60)
print("示例2: Haystack 向量检索")
print("=" * 60)

# 创建带嵌入的文档存储
vector_store = InMemoryDocumentStore()

# 创建文档嵌入器
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

# 嵌入并存储文档
docs_with_embeddings = doc_embedder.run(documents=documents)
vector_store.write_documents(docs_with_embeddings['documents'])
print(f"\n已嵌入并存储 {len(documents)} 个文档")

# 创建向量检索器
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
text_embedder.warm_up()

vector_retriever = InMemoryEmbeddingRetriever(document_store=vector_store)

# 测试向量检索
query = "什么是编程语言？"
print(f"\n查询: {query}")

# 嵌入查询
query_embedding = text_embedder.run(text=query)
results = vector_retriever.run(query_embedding=query_embedding['embedding'], top_k=2)

print("相关文档:")
for i, doc in enumerate(results['documents'], 1):
    print(f"  {i}. {doc.content}")


# ========== 示例3: RAG Pipeline ==========
print("\n" + "=" * 60)
print("示例3: Haystack RAG Pipeline")
print("=" * 60)

# 创建RAG pipeline
rag_pipeline = Pipeline()

# 添加组件
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", vector_retriever)
rag_pipeline.add_component("prompt_builder", PromptBuilder(template="""
基于以下上下文回答问题：

上下文：
{% for document in documents %}
{{ document.content }}
{% endfor %}

问题：{{ question }}

回答：
"""))

if api_key:
    rag_pipeline.add_component("llm", OpenAIGenerator(api_key=Secret.from_token(api_key), model="gpt-3.5-turbo"))
    rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "prompt_builder.documents")
    rag_pipeline.connect("prompt_builder", "llm")
    
    # 运行RAG pipeline
    questions = [
        "Python是谁创建的？",
        "机器学习是什么？",
        "RAG技术有什么优势？"
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        result = rag_pipeline.run({
            "text_embedder": {"text": question},
            "prompt_builder": {"question": question}
        })
        
        print(f"回答: {result['llm']['replies'][0]}")
else:
    print("\n⚠️ 未配置API Key，跳过LLM生成步骤")
    print("配置OPENAI_API_KEY环境变量后可运行完整RAG流程")


# ========== 示例4: 高级Pipeline编排 ==========
print("\n" + "=" * 60)
print("示例4: Haystack 高级Pipeline编排")
print("=" * 60)

# 创建多阶段pipeline
advanced_pipeline = Pipeline()

# 添加多个检索器
bm25_retriever = InMemoryBM25Retriever(document_store=document_store)

advanced_pipeline.add_component("bm25_retriever", bm25_retriever)
advanced_pipeline.add_component("vector_retriever", vector_retriever)

# 注意：实际使用中可能需要合并多个检索结果
# 这里简化演示

print("\nPipeline组件:")
print("  - BM25检索器（关键词匹配）")
print("  - 向量检索器（语义匹配）")
print("  - 提示构建器")
print("  - LLM生成器")


# ========== 示例5: 文档过滤 ==========
print("\n" + "=" * 60)
print("示例5: Haystack 文档过滤")
print("=" * 60)

# 创建带元数据的文档
meta_documents = [
    Document(content="Python教程", meta={"category": "programming", "level": "beginner"}),
    Document(content="AI基础", meta={"category": "ai", "level": "beginner"}),
    Document(content="深度学习进阶", meta={"category": "ai", "level": "advanced"}),
    Document(content="Web开发", meta={"category": "web", "level": "intermediate"}),
]

meta_store = InMemoryDocumentStore()
meta_store.write_documents(meta_documents)

# 创建带过滤的检索器
filtered_retriever = InMemoryBM25Retriever(document_store=meta_store)

# 测试过滤检索
filters = {"field": "meta.category", "operator": "==", "value": "ai"}
print(f"\n使用过滤器: {filters}")

results = filtered_retriever.run(query="AI", filters=filters, top_k=5)
print("检索结果:")
for doc in results['documents']:
    print(f"  - {doc.content} (类别: {doc.meta.get('category')}, 难度: {doc.meta.get('level')})")


# ========== 示例6: 批量处理 ==========
print("\n" + "=" * 60)
print("示例6: Haystack 批量处理")
print("=" * 60)

# 批量查询
batch_queries = ["Python", "AI", "机器学习", "深度学习"]

print("\n批量查询:")
for query in batch_queries:
    results = retriever.run(query=query, top_k=1)
    if results['documents']:
        print(f"  {query}: {results['documents'][0].content[:50]}...")


# ========== 示例7: 性能优化 ==========
print("\n" + "=" * 60)
print("示例7: Haystack 性能优化技巧")
print("=" * 60)

optimization_tips = """
Haystack性能优化建议：

1. 文档预处理：
   - 使用合适的chunk大小（通常200-500 tokens）
   - 添加有意义的元数据
   - 清理无关内容

2. 检索优化：
   - 根据场景选择BM25或向量检索
   - 调整top_k参数平衡准确性和速度
   - 使用混合检索（Hybrid Search）

3. 缓存策略：
   - 缓存频繁查询的结果
   - 缓存文档嵌入向量
   - 使用Redis等外部缓存

4. 批处理：
   - 批量嵌入文档
   - 批量查询减少网络开销
   - 异步处理提高吞吐量

5. 资源管理：
   - 定期清理无用文档
   - 监控内存使用
   - 使用持久化存储
"""

print(optimization_tips)


print("\n" + "=" * 60)
print("Haystack 示例完成！")
print("=" * 60)
