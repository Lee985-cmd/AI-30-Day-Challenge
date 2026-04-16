"""
意图识别 Agent
"""
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
import os


class IntentAgent:
    """意图识别 Agent"""
    
    def __init__(self, api_key: str = None):
        """
        初始化意图识别 Agent
        
        Args:
            api_key: 阿里云 DashScope API Key（可选，优先使用环境变量）
        """
        # 优先使用环境变量
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量或传入 api_key 参数")
        
        os.environ["DASHSCOPE_API_KEY"] = self.api_key
        
        # 使用通义千问模型
        self.llm = ChatTongyi(
            model="qwen-turbo",  # 可使用 qwen-turbo, qwen-plus, qwen-max
            temperature=0
        )
        
        # 意图分类提示词
        prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""你是一个电商客服意图识别专家。请判断用户的意图属于哪一类。

可选类别：
- 售前咨询：询问产品信息、价格、规格等
- 售后问题：退货、换货、维修、退款
- 物流查询：发货时间、物流进度、修改地址
- 发票问题：开票、发票类型、邮寄发票
- 投诉建议：产品质量投诉、服务投诉、建议
- 其他：不属于以上类别

用户输入：{user_input}

请只输出类别名称，不要解释。
类别："""
        )
        
        self.chain = LLMChain(llm=self.llm, prompt=prompt)
    
    def classify(self, user_input: str) -> str:
        """
        识别用户意图
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            意图类别
        """
        # 使用 invoke 替代 run（新版本 API）
        result = self.chain.invoke({"user_input": user_input})
        
        # 调试：打印返回结果类型和内容
        print(f"[DEBUG IntentAgent] result type: {type(result)}")
        print(f"[DEBUG IntentAgent] result: {result}")
        
        # invoke 返回字典，需要提取 text 字段
        if isinstance(result, dict):
            # 尝试多个可能的键
            intent = result.get("text", result.get("output", result.get("result", "")))
        else:
            intent = str(result)
        
        return intent.strip()


if __name__ == "__main__":
    # 测试代码
    api_key = os.getenv("OPENAI_API_KEY", "your-api-key")
    agent = IntentAgent(api_key=api_key)
    
    test_cases = [
        "这个产品多少钱？",
        "我想退货",
        "什么时候发货？",
        "能开发票吗？",
        "你们的产品质量太差了"
    ]
    
    print("意图识别测试：")
    print("=" * 50)
    for case in test_cases:
        intent = agent.classify(case)
        print(f"输入：{case}")
        print(f"意图：{intent}\n")
