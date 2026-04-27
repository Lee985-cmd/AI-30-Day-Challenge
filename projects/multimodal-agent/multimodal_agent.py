"""
多模态智能客服 Agent
支持图像 + 文本的智能对话系统
"""

import os
import base64
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from PIL import Image
import io


class MultimodalAgent:
    """多模态 Agent - 支持图像和文本理解"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        初始化多模态 Agent
        
        Args:
            api_key: API密钥（可选，优先使用环境变量）
        """
        # 使用本地模型或云端API
        local_llm_url = os.getenv("LOCAL_LLM_URL")
        
        if local_llm_url:
            # 使用本地 OpenAI 兼容模型
            self.llm = ChatOpenAI(
                model="qwen-vl-plus",  # 通义千问视觉模型
                openai_api_base=local_llm_url,
                openai_api_key="not-needed",
                temperature=0.3
            )
            self.use_local_model = True
        else:
            # 使用 OpenAI GPT-4V
            openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("请设置 OPENAI_API_KEY 环境变量或传入 api_key")
            
            self.llm = ChatOpenAI(
                model="gpt-4-vision-preview",
                openai_api_key=openai_api_key,
                temperature=0.3
            )
            self.use_local_model = False
        
        # 系统提示词
        self.system_prompt = """你是一个专业的智能客服助手，具备图像识别能力。

你的职责：
1. **图像理解**：仔细分析用户上传的图片，识别产品、问题、故障等
2. **问题诊断**：根据图片和文字描述，判断用户遇到的问题
3. **解决方案**：提供清晰、可操作的解决步骤
4. **友好沟通**：用温暖、专业的语气与用户交流

回答要求：
- 先描述你从图片中看到的内容
- 然后分析问题可能的原因
- 最后给出具体的解决方案
- 如果无法确定，诚实地告诉用户并提供建议

保持专业、耐心、有帮助的态度。"""
        
        # 对话历史
        self.conversation_history: List[Dict] = []
    
    def encode_image(self, image_path: str) -> str:
        """
        将图片编码为 base64
        
        Args:
            image_path: 图片路径
            
        Returns:
            base64 编码的字符串
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def _prepare_image_content(self, image_path: str, text: str) -> list:
        """
        准备图片内容（兼容不同模型）
        
        Args:
            image_path: 图片路径
            text: 文本内容
            
        Returns:
            消息内容列表
        """
        # 编码图片
        base64_image = self.encode_image(image_path)
        
        if self.use_local_model:
            # 本地模型可能支持不同的格式
            # 尝试使用标准 OpenAI 格式
            return [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        else:
            # OpenAI GPT-4V 格式
            return [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
    
    def analyze_image(self, image_path: str, question: str = "请描述这张图片") -> str:
        """
        分析单张图片
        
        Args:
            image_path: 图片路径
            question: 问题描述
            
        Returns:
            AI 的回答
        """
        # 构建消息
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=self._prepare_image_content(image_path, question))
        ]
        
        # 调用模型
        response = self.llm.invoke(messages)
        
        return response.content
    
    def chat_with_image(self, image_path: str, user_message: str) -> str:
        """
        带图片的对话
        
        Args:
            image_path: 图片路径
            user_message: 用户消息
            
        Returns:
            AI 的回答
        """
        # 构建消息（包含历史对话）
        messages = [SystemMessage(content=self.system_prompt)]
        
        # 添加历史对话
        for msg in self.conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # 添加当前消息（带图片）
        messages.append(HumanMessage(content=self._prepare_image_content(image_path, user_message)))
        
        # 调用模型
        response = self.llm.invoke(messages)
        
        # 保存对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        return response.content
    
    def chat_text_only(self, user_message: str) -> str:
        """
        纯文本对话（无图片）
        
        Args:
            user_message: 用户消息
            
        Returns:
            AI 的回答
        """
        # 构建消息
        messages = [SystemMessage(content=self.system_prompt)]
        
        # 添加历史对话
        for msg in self.conversation_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        # 添加当前消息
        messages.append(HumanMessage(content=user_message))
        
        # 调用模型
        response = self.llm.invoke(messages)
        
        # 保存对话历史
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": response.content
        })
        
        return response.content
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
    
    def get_product_info(self, image_path: str) -> Dict:
        """
        从图片中提取产品信息
        
        Args:
            image_path: 产品图片路径
            
        Returns:
            产品信息字典
        """
        question = """请仔细分析这张产品图片，提取以下信息：
1. 产品名称/类型
2. 品牌（如果可见）
3. 颜色
4. 明显的特征或标识
5. 可能的用途

请以 JSON 格式返回，例如：
{
  "product_name": "xxx",
  "brand": "xxx",
  "color": "xxx",
  "features": ["xxx", "xxx"],
  "usage": "xxx"
}"""
        
        response = self.analyze_image(image_path, question)
        
        # 尝试解析 JSON
        import json
        import re
        
        # 提取 JSON 部分
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                product_info = json.loads(json_match.group())
                return product_info
            except:
                pass
        
        # 如果解析失败，返回原始文本
        return {"raw_response": response}
    
    def diagnose_problem(self, image_path: str, description: str = "") -> Dict:
        """
        诊断产品问题
        
        Args:
            image_path: 问题产品图片
            description: 文字描述（可选）
            
        Returns:
            诊断结果
        """
        question = f"""你是一个专业的产品技术支持工程师。

请分析这张图片，并结合以下描述（如果有）：
{description if description else "无额外描述"}

请提供：
1. **问题识别**：你看到了什么问题？
2. **可能原因**：导致这个问题的可能原因有哪些？
3. **解决方案**：具体的解决步骤（分点列出）
4. **预防建议**：如何避免类似问题再次发生

请用中文回答，保持专业和清晰。"""
        
        response = self.analyze_image(image_path, question)
        
        return {
            "diagnosis": response,
            "image_path": image_path,
            "description": description
        }


# 测试代码
if __name__ == "__main__":
    print("多模态智能客服 Agent")
    print("=" * 50)
    
    # 创建 Agent
    agent = MultimodalAgent()
    
    print("\n✅ Agent 初始化成功！")
    print("\n功能说明：")
    print("1. analyze_image() - 分析单张图片")
    print("2. chat_with_image() - 带图片对话")
    print("3. chat_text_only() - 纯文本对话")
    print("4. get_product_info() - 提取产品信息")
    print("5. diagnose_problem() - 诊断产品问题")
    print("\n💡 运行 streamlit_app.py 启动 Web 界面")
