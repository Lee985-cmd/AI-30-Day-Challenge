"""
对话管理 Agent
"""
from langchain_community.chat_models import ChatTongyi
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from typing import Dict, List
import os


class DialogueAgent:
    """对话管理 Agent"""
    
    def __init__(self, api_key: str = None, knowledge_base=None):
        """
        初始化对话管理 Agent
        
        Args:
            api_key: 阿里云 DashScope API Key（可选，优先使用环境变量）
            knowledge_base: KnowledgeBase 实例
        """
        # 优先使用环境变量
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")
        
        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        self.kb = knowledge_base
        
        # 初始化大模型 - 使用通义千问
        self.llm = ChatTongyi(
            model="qwen-plus",  # 对话场景推荐使用 qwen-plus
            temperature=0.7
        )
        
        # 多用户对话历史存储 {user_id: [(question, answer), ...]}
        self.conversation_histories: Dict[str, List[tuple]] = {}
        
        # 自定义提示词
        qa_prompt = PromptTemplate(
            template="""你是一个专业的电商客服助手。请根据以下上下文回答用户问题。

规则：
1. 只使用提供的上下文信息回答，不要编造
2. 如果上下文中没有答案，说“抱歉，我暂时无法回答这个问题，建议您联系人工客服”
3. 语气友好、专业、简洁
4. 不要提及“上下文”这个词
5. 结合聊天历史，保持对话连贯性

上下文：
{context}

用户问题：{question}

客服回答："""
        )
        
        # 创建对话链（不使用内置记忆，手动管理）
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=knowledge_base.vector_db.as_retriever(search_kwargs={"k": 3}),
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            return_source_documents=True
        )
    
    def chat(self, user_input: str, user_id: str = "default_user") -> dict:
        """
        处理用户对话
        
        Args:
            user_input: 用户输入文本
            user_id: 用户 ID（用于多轮对话）
            
        Returns:
            包含回答、来源、置信度的字典
        """
        # 获取该用户的对话历史
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
        
        history = self.conversation_histories[user_id]
        
        # 格式化聊天历史为列表格式 [(human_message, ai_message), ...]
        chat_history_list = self._format_chat_history_list(history)
        
        # 使用 invoke 调用 Chain
        result = self.chain.invoke({
            "question": user_input,
            "chat_history": chat_history_list  # 传入列表格式
        })
        
        answer = result["answer"]
        source_docs = result.get("source_documents", [])
        
        # 提取来源文档
        sources = []
        for doc in source_docs:
            sources.append(doc.page_content[:100])
        
        # 保存对话历史（最多保留最近 10 轮）
        history.append((user_input, answer))
        if len(history) > 10:
            self.conversation_histories[user_id] = history[-10:]
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": self._calculate_confidence(answer, sources)
        }
    
    def _format_chat_history_list(self, history: List[tuple]) -> List[tuple]:
        """
        格式化聊天历史为列表格式
        
        Args:
            history: [(question, answer), ...]
            
        Returns:
            列表格式的聊天历史 [(human_msg, ai_msg), ...]
        """
        # 只取最近 5 轮对话
        recent_history = history[-5:] if len(history) > 5 else history
        return recent_history
    
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
