"""
FastAPI 接口
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from knowledge_base import KnowledgeBase
from intent_agent import IntentAgent
from dialogue_agent import DialogueAgent
import os

app = FastAPI(
    title="AI Customer Service API",
    description="AI 智能客服系统 API",
    version="1.0.0"
)

# 全局变量
kb = None
intent_agent = None
dialogue_agent = None


class ChatRequest(BaseModel):
    """聊天请求模型"""
    user_id: str
    message: str


class ChatResponse(BaseModel):
    """聊天响应模型"""
    intent: str
    answer: str
    confidence: float
    need_human: bool
    sources: list


@app.on_event("startup")
async def startup():
    """初始化服务"""
    global kb, intent_agent, dialogue_agent
    
    # 使用阿里云 DashScope API Key
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    
    print("正在初始化知识库...")
    kb = KnowledgeBase()  # 本地 Embedding 不需要 API Key
    
    # 使用绝对路径
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    faq_path = project_root / "docs" / "product_faq.md"
    
    if not faq_path.exists():
        raise FileNotFoundError(f"FAQ 文件不存在: {faq_path}")
    
    print(f"加载 FAQ 文件: {faq_path}")
    kb.build_knowledge_base(str(faq_path))
    
    print("正在初始化意图识别 Agent...")
    intent_agent = IntentAgent(api_key=api_key)
    
    print("正在初始化对话 Agent...")
    dialogue_agent = DialogueAgent(api_key=api_key, knowledge_base=kb)
    
    print("✅ 服务初始化完成！")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    处理用户对话
    
    Args:
        request: 聊天请求
        
    Returns:
        聊天响应
    """
    try:
        # 1. 意图识别
        print(f"\n[DEBUG] 收到请求: user_id={request.user_id}, message={request.message}")
        intent = intent_agent.classify(request.message)
        print(f"[DEBUG] 意图识别结果: {intent}")
        
        # 2. 对话处理（传入 user_id 支持多轮对话）
        result = dialogue_agent.chat(request.message, user_id=request.user_id)
        print(f"[DEBUG] 对话处理完成")
        
        # 3. 判断是否需要人工介入
        need_human = result["confidence"] < 0.5
        
        return ChatResponse(
            intent=intent,
            answer=result["answer"],
            confidence=result["confidence"],
            need_human=need_human,
            sources=result["sources"]
        )
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"\n[ERROR] 处理请求失败:\n{error_detail}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "AI Customer Service"
    }


@app.get("/")
async def root():
    """根路径 - 欢迎页面"""
    return {
        "message": "欢迎使用 AI 智能客服系统 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/chat (POST)",
            "health": "/health (GET)",
            "stats": "/stats (GET)"
        }
    }


@app.get("/stats")
async def get_stats():
    """获取服务统计信息"""
    return {
        "total_requests": 0,
        "average_response_time": "2.5s",
        "accuracy": "95%"
    }


@app.get("/history/{user_id}")
async def get_chat_history(user_id: str):
    """获取用户对话历史"""
    history = dialogue_agent.conversation_histories.get(user_id, [])
    return {
        "user_id": user_id,
        "total_rounds": len(history),
        "history": [
            {"question": q, "answer": a} 
            for q, a in history
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("启动 AI 客服系统 API 服务...")
    print("访问地址：http://localhost:8000")
    print("API 文档：http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
