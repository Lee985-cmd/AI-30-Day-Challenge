"""
知识库模块 - RAG 实现
"""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


class KnowledgeBase:
    """知识库管理类"""
    
    def __init__(self, api_key: str = None):
        """
        初始化知识库
        
        Args:
            api_key: API Key（可选，本地 Embedding 不需要）
        """
        # 使用本地 Embedding 模型（完全免费，无需 API Key）
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # 使用 CPU，如有 GPU 可改为 'cuda'
        )
        self.vector_db = None
    
    def build_knowledge_base(self, doc_path: str):
        """
        构建知识库
        
        Args:
            doc_path: 文档路径
        """
        print("正在加载文档...")
        # 1. 加载文档
        loader = TextLoader(doc_path, encoding='utf-8')
        documents = loader.load()
        
        print("正在切分文档...")
        # 2. 切分文档（每段 500 字，重叠 50 字）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = text_splitter.split_documents(documents)
        
        print("正在创建向量索引...")
        # 3. 创建向量数据库
        self.vector_db = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        print(f"✅ 知识库构建完成！共 {len(texts)} 个文档块")
    
    def search(self, query: str, k: int = 3) -> str:
        """
        检索最相关的文档
        
        Args:
            query: 查询文本
            k: 返回最相关的 k 个文档
            
        Returns:
            合并后的文档内容
        """
        if not self.vector_db:
            raise ValueError("知识库未初始化")
        
        # 检索最相关的 k 个文档块
        results = self.vector_db.similarity_search(query, k=k)
        
        # 合并结果
        context = "\n\n".join([doc.page_content for doc in results])
        return context


if __name__ == "__main__":
    # 测试代码
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
    kb = KnowledgeBase(api_key=api_key)
    kb.build_knowledge_base("../docs/product_faq.md")
    
    # 测试检索
    query = "怎么退货？"
    context = kb.search(query)
    print(f"\n检索结果：\n{context}")
