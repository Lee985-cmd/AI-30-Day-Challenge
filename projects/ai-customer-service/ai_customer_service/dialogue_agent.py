"""
对话管理 Agent
"""
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os


class DialogueAgent:
    """对话管理 Agent"""
    
    def __init__(self, api_key: str, knowledge_base):
        """
        初始化对话管理 Agent
        
        Args:
            api_key: OpenAI API Key
            knowledge_base: KnowledgeBase 实例
        """
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        self.kb = knowledge_base
        
        # 初始化大模型
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        
        # 对话记忆
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 自定义提示词
        qa_prompt = PromptTemplate(
            template="""你是一个专业的电商客服助手。请根据以下上下文回答用户问题。

规则：
1. 只使用提供的上下文信息回答，不要编造
2. 如果上下文中没有答案，说"抱歉，我暂时无法回答这个问题，建议您联系人工客服"
3. 语气友好、专业、简洁
4. 不要提及"上下文"这个词

上下文：
{context}

聊天历史：
{chat_history}

用户问题：{question}

客服回答："""
        )
        
        # 创建对话链
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=knowledge_base.vector_db.as_retriever(search_kwargs={"k": 3}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
    
    def chat(self, user_input: str) -> dict:
        """
        处理用户对话
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            包含回答、来源、置信度的字典
        """
        result = self.chain({"question": user_input})
        
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        
        # 提取来源文档
        sources = []
        for doc in source_docs:
            sources.append(doc.page_content[:100])
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": self._calculate_confidence(answer, sources)
        }
    
    def _calculate_confidence(self, answer: str, sources: list) -> float:
        """
        计算回答置信度（简化版）
        
        Args:
            answer: 回答文本
            sources: 来源文档列表
            
        Returns:
            置信度（0-1）
        """
        if not sources:
            return 0.0
        # 如果有来源文档，置信度高
        return min(0.9, len(sources) * 0.3)


if __name__ == "__main__":
    # 测试代码
    from knowledge_base import KnowledgeBase
    
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
    
    print("初始化知识库...")
    kb = KnowledgeBase(api_key=api_key)
    kb.build_knowledge_base("../docs/product_faq.md")
    
    print("\n初始化对话 Agent...")
    agent = DialogueAgent(api_key=api_key, knowledge_base=kb)
    
    # 多轮对话测试
    questions = [
        "怎么退货？",
        "退货有时间限制吗？",
        "运费谁承担？"
    ]
    
    print("\n对话测试：")
    print("=" * 50)
    for q in questions:
        print(f"\n用户：{q}")
        result = agent.chat(q)
        print(f"客服：{result['answer']}")
        print(f"置信度：{result['confidence']:.2f}")
