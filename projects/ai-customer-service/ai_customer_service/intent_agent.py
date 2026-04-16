"""
意图识别 Agent
"""
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os


class IntentAgent:
    """意图识别 Agent"""
    
    def __init__(self, api_key: str):
        """
        初始化意图识别 Agent
        
        Args:
            api_key: OpenAI API Key
        """
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
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
        intent = self.chain.run(user_input=user_input).strip()
        return intent


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
