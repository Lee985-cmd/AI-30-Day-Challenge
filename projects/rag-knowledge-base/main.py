"""
RAG 知识库问答系统 - 主程序
"""

import argparse
import logging
from pathlib import Path
import sys

from config import LOG_LEVEL, LOG_FILE
from document_loader import load_and_split
from vector_store import create_vector_store, load_vector_store
from retriever import create_retriever
from qa_chain import create_qa_chain

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def build_knowledge_base(rebuild: bool = False):
    """
    构建知识库
    
    Args:
        rebuild: 是否重建
    """
    print("\n" + "="*60)
    print("🔨 构建知识库")
    print("="*60)
    
    # 1. 加载文档
    print("\n📂 步骤 1/3: 加载文档...")
    chunks = load_and_split()
    
    if not chunks:
        print("❌ 没有找到文档，请先在 documents/ 目录下放置文档")
        print("\n支持的格式: PDF, TXT, Markdown, DOCX")
        return False
    
    # 2. 创建向量数据库
    print("\n📊 步骤 2/3: 创建向量数据库...")
    vector_store = create_vector_store(chunks, rebuild=rebuild)
    
    # 3. 获取统计信息
    print("\n📈 步骤 3/3: 统计信息...")
    stats = vector_store.get_statistics()
    
    print("\n✅ 知识库构建完成！")
    print("\n📊 统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True


def start_chat():
    """启动交互式问答"""
    print("\n" + "="*60)
    print("💬 启动问答系统")
    print("="*60)
    
    # 加载向量数据库
    print("\n📊 加载知识库...")
    manager = load_vector_store()
    
    if not manager or not manager.vectorstore:
        print("❌ 知识库不存在，请先构建知识库")
        print("💡 运行: python main.py --mode build")
        return
    
    # 创建检索器
    retriever = create_retriever(manager.vectorstore)
    
    # 创建问答链
    qa_chain = create_qa_chain(retriever)
    
    # 启动聊天
    qa_chain.chat()


def ask_single_question(question: str):
    """单次问答"""
    print("\n" + "="*60)
    print("❓ 问答模式")
    print("="*60)
    print(f"\n问题: {question}\n")
    
    # 加载向量数据库
    print("📊 加载知识库...")
    manager = load_vector_store()
    
    if not manager or not manager.vectorstore:
        print("❌ 知识库不存在，请先构建知识库")
        return
    
    # 创建检索器
    retriever = create_retriever(manager.vectorstore)
    
    # 创建问答链
    qa_chain = create_qa_chain(retriever)
    
    # 回答问题
    print("⏳ 正在生成回答...\n")
    result = qa_chain.ask(question)
    
    # 显示结果
    print("="*60)
    print("💡 回答:")
    print("="*60)
    print(result['answer'])
    
    if result['sources']:
        print("\n" + "-"*60)
        print("📚 参考来源:")
        print("-"*60)
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[{i}] {source['filename']}")
            print(f"    {source['content']}...")
    
    print("\n" + "="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='RAG 知识库问答系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 构建知识库
  python main.py --mode build
  
  # 交互式问答
  python main.py --mode chat
  
  # 单次问答
  python main.py --mode ask --question "如何安装？"
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['build', 'chat', 'ask'],
        default='chat',
        help='运行模式: build(构建知识库), chat(交互式问答), ask(单次问答)'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='单次问答的问题（仅在 ask 模式下使用）'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='重建知识库（删除旧数据）'
    )
    
    args = parser.parse_args()
    
    # 根据模式执行
    if args.mode == 'build':
        success = build_knowledge_base(rebuild=args.rebuild)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'chat':
        start_chat()
    
    elif args.mode == 'ask':
        if not args.question:
            parser.error("ask 模式需要 --question 参数")
        ask_single_question(args.question)


if __name__ == '__main__':
    main()
