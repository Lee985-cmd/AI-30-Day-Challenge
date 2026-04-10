"""
文档加载和解析模块
支持 PDF、TXT、Markdown、DOCX 等多种格式
"""

import os
from pathlib import Path
from typing import List
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import logging

from config import (
    DOCS_DIR,
    SUPPORTED_FORMATS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SPLITTER_METHOD
)

logger = logging.getLogger(__name__)


class DocumentLoader:
    """文档加载器"""
    
    def __init__(self, docs_dir: Path = None):
        self.docs_dir = docs_dir or DOCS_DIR
        self.documents = []
    
    def load_all_documents(self) -> List[Document]:
        """
        加载目录下的所有文档
        
        Returns:
            List[Document]: 文档列表
        """
        logger.info(f"📂 开始加载文档目录: {self.docs_dir}")
        
        if not self.docs_dir.exists():
            logger.warning(f"⚠️  文档目录不存在: {self.docs_dir}")
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            return []
        
        all_docs = []
        
        # 遍历目录
        for file_path in self.docs_dir.rglob('*'):
            if not file_path.is_file():
                continue
            
            # 检查文件格式
            if file_path.suffix.lower() not in SUPPORTED_FORMATS:
                continue
            
            try:
                logger.info(f"📄 加载文档: {file_path.name}")
                docs = self._load_single_document(file_path)
                all_docs.extend(docs)
                logger.info(f"✅ 成功加载 {len(docs)} 个文档块")
                
            except Exception as e:
                logger.error(f"❌ 加载失败 {file_path.name}: {str(e)}")
                continue
        
        self.documents = all_docs
        logger.info(f"🎉 总共加载 {len(all_docs)} 个文档块")
        
        return all_docs
    
    def _load_single_document(self, file_path: Path) -> List[Document]:
        """
        加载单个文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            List[Document]: 文档列表
        """
        suffix = file_path.suffix.lower()
        
        # 根据格式选择加载器
        if suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
        elif suffix == '.txt':
            loader = TextLoader(str(file_path), encoding='utf-8')
        elif suffix == '.md':
            loader = UnstructuredMarkdownLoader(str(file_path))
        elif suffix == '.docx':
            loader = Docx2txtLoader(str(file_path))
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
        
        # 加载文档
        docs = loader.load()
        
        # 添加元数据
        for doc in docs:
            doc.metadata['source'] = str(file_path)
            doc.metadata['filename'] = file_path.name
        
        return docs
    
    def split_documents(self, documents: List[Document] = None) -> List[Document]:
        """
        将文档切分成文本块
        
        Args:
            documents: 文档列表，如果为 None 则使用已加载的文档
            
        Returns:
            List[Document]: 切分后的文档块列表
        """
        if documents is None:
            documents = self.documents
        
        if not documents:
            logger.warning("⚠️  没有文档可切分")
            return []
        
        logger.info(f"✂️  开始切分文档 (CHUNK_SIZE={CHUNK_SIZE}, OVERLAP={CHUNK_OVERLAP})")
        
        # 创建文本分割器
        if SPLITTER_METHOD == 'recursive':
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
            )
        else:
            # 固定长度切分
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
        
        # 切分文档
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"✅ 切分完成: {len(documents)} 个文档 → {len(chunks)} 个文本块")
        
        return chunks


def load_and_split(docs_dir: Path = None) -> List[Document]:
    """
    便捷函数：加载并切分文档
    
    Args:
        docs_dir: 文档目录路径
        
    Returns:
        List[Document]: 切分后的文档块列表
    """
    loader = DocumentLoader(docs_dir)
    documents = loader.load_all_documents()
    chunks = loader.split_documents(documents)
    return chunks


if __name__ == '__main__':
    # 测试
    import logging
    logging.basicConfig(level=logging.INFO)
    
    chunks = load_and_split()
    
    print(f"\n📊 统计信息:")
    print(f"  文本块数量: {len(chunks)}")
    if chunks:
        print(f"  第一个文本块:")
        print(f"    来源: {chunks[0].metadata.get('source', '未知')}")
        print(f"    长度: {len(chunks[0].page_content)} 字符")
        print(f"    内容预览: {chunks[0].page_content[:100]}...")
