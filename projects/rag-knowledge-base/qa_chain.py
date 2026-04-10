"""
问答链模块
组合检索器和 LLM 生成回答
"""

import logging
from typing import Optional
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    USE_LOCAL_MODEL,
    LOCAL_MODEL_URL,
    LOCAL_MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    QA_TEMPLATE
)

logger = logging.getLogger(__name__)


class QAChain:
    """问答链"""
    
    def __init__(self, retriever):
        """
        初始化问答链
        
        Args:
            retriever: 检索器实例
        """
        self.retriever = retriever
        self.llm = None
        self.qa_chain = None
        
        self._create_llm()
        self._create_qa_chain()
    
    def _create_llm(self):
        """创建 LLM 实例"""
        logger.info("🤖 创建 LLM 模型...")
        
        if USE_LOCAL_MODEL:
            # 使用本地模型
            from langchain.llms import ChatGLM
            
            self.llm = ChatGLM(
                endpoint_url=LOCAL_MODEL_URL,
                max_token=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            logger.info(f"✅ 使用本地模型: {LOCAL_MODEL_NAME}")
        else:
            # 使用 OpenAI
            self.llm = OpenAI(
                openai_api_key=OPENAI_API_KEY,
                model_name=OPENAI_MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            logger.info(f"✅ 使用 OpenAI 模型: {OPENAI_MODEL}")
    
    def _create_qa_chain(self):
        """创建问答链"""
        logger.info("🔗 创建问答链...")
        
        # 创建 prompt 模板
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=QA_TEMPLATE
        )
        
        # 创建 RetrievalQA 链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # 将所有文档合并到一个 prompt
            retriever=self.retriever.vectorstore.as_retriever(
                search_kwargs={'k': self.retriever.top_k}
            ),
            chain_type_kwargs={
                'prompt': prompt_template,
                'document_prompt': PromptTemplate(
                    input_variables=["page_content", "source"],
                    template="来源: {source}\n\n内容:\n{page_content}"
                )
            },
            return_source_documents=True  # 返回来源文档
        )
        
        logger.info("✅ 问答链创建完成")
    
    def ask(self, question: str, return_sources: bool = True) -> dict:
        """
        回答问题
        
        Args:
            question: 问题
            return_sources: 是否返回来源文档
            
        Returns:
            dict: 包含回答和来源的字典
        """
        logger.info(f"💬 提问: {question}")
        
        try:
            # 执行问答链
            result = self.qa_chain({"query": question})
            
            answer = result['result']
            source_docs = result.get('source_documents', [])
            
            response = {
                'question': question,
                'answer': answer,
                'sources': []
            }
            
            # 提取来源信息
            if return_sources and source_docs:
                for doc in source_docs:
                    source_info = {
                        'content': doc.page_content[:200],  # 只取前 200 字符
                        'source': doc.metadata.get('source', '未知'),
                        'filename': doc.metadata.get('filename', '未知')
                    }
                    response['sources'].append(source_info)
            
            logger.info(f"✅ 回答生成完成 (长度: {len(answer)} 字符)")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 回答生成失败: {str(e)}")
            return {
                'question': question,
                'answer': f"抱歉，回答生成时出现错误: {str(e)}",
                'sources': []
            }
    
    def chat(self):
        """交互式聊天模式"""
        print("\n" + "="*60)
        print("🤖 RAG 知识库问答系统")
        print("="*60)
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'clear' 清空屏幕")
        print("="*60 + "\n")
        
        while True:
            try:
                # 获取用户输入
                question = input("\n❓ 你的问题: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 再见！")
                    break
                
                if question.lower() == 'clear':
                    print("\033[H\033[J")  # 清屏
                    continue
                
                # 回答问题
                print("\n⏳ 正在思考...")
                result = self.ask(question)
                
                # 显示回答
                print("\n" + "="*60)
                print("💡 回答:")
                print("="*60)
                print(result['answer'])
                
                # 显示来源
                if result['sources']:
                    print("\n" + "-"*60)
                    print("📚 参考来源:")
                    print("-"*60)
                    for i, source in enumerate(result['sources'], 1):
                        print(f"\n[{i}] {source['filename']}")
                        print(f"    {source['content']}...")
                
                print("="*60)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                logger.error(f"❌ 聊天出错: {str(e)}")
                print(f"\n❌ 错误: {str(e)}")


def create_qa_chain(retriever) -> QAChain:
    """
    便捷函数：创建问答链
    
    Args:
        retriever: 检索器实例
        
    Returns:
        QAChain: 问答链实例
    """
    return QAChain(retriever)


if __name__ == '__main__':
    # 测试
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from vector_store import load_vector_store
    from retriever import create_retriever
    
    # 加载向量数据库
    manager = load_vector_store()
    
    if manager and manager.vectorstore:
        # 创建检索器
        retriever = create_retriever(manager.vectorstore)
        
        # 创建问答链
        qa_chain = create_qa_chain(retriever)
        
        # 测试单个问题
        question = "如何安装这个软件？"
        result = qa_chain.ask(question)
        
        print(f"\n📝 问题: {result['question']}")
        print(f"\n💡 回答: {result['answer']}")
        
        if result['sources']:
            print(f"\n📚 来源数量: {len(result['sources'])}")
