# 多模态支持说明

## ⚠️ 当前状态

经过测试，你的本地模型（`http://61.49.53.5:30001/v1`）**不支持真正的多模态图片识别**。

### 测试结果

```
✅ API调用成功
❌ 模型返回："抱歉，我没有看到您上传的图片"
```

这说明：
- 模型API接口正常
- 但模型本身不具备视觉理解能力
- 或者使用的模型名称不正确

---

## 🔍 原因分析

### 可能的原因

1. **模型类型不对**
   - 当前使用的模型可能不是视觉模型（如 qwen-vl, yi-vl 等）
   - 需要确认本地部署的是什么模型

2. **模型名称配置错误**
   - 代码中使用的是 `"qwen-vl-plus"`
   - 实际部署的模型名称可能不同

3. **模型不支持多模态**
   - 某些模型只支持文本，不支持图片

---

## 💡 解决方案

### 方案 1: 使用"图片描述+文本"模式（当前推荐）

**工作原理：**
- 用户上传图片（作为参考）
- 用户手动描述图片内容
- AI基于文字描述提供解答

**优点：**
- ✅ 立即可用
- ✅ 不依赖视觉模型
- ✅ 成本低

**缺点：**
- ❌ 需要用户手动描述
- ❌ 无法自动识别图片

**使用方法：**

1. 上传图片（作为参考）
2. 在文本框中描述图片内容，例如：
   ```
   图片中是一台笔记本电脑，屏幕有裂纹，无法开机。
   请问如何维修？
   ```

3. AI会根据你的描述提供解决方案

---

### 方案 2: 更换为支持多模态的模型

#### 选项 A: 使用通义千问视觉模型

如果你使用的是阿里云的模型服务：

```python
# 修改 multimodal_agent.py 中的模型名称
self.llm = ChatOpenAI(
    model="qwen-vl-max",  # 或 qwen-vl-plus
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key="your-api-key",
    temperature=0.3
)
```

**获取API Key:**
- 访问：https://dashscope.console.aliyun.com/
- 创建 API Key
- 充值（有免费额度）

#### 选项 B: 使用 OpenAI GPT-4V

```python
# 设置环境变量
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-xxx", "User")

# 清除 LOCAL_LLM_URL
[System.Environment]::SetEnvironmentVariable("LOCAL_LLM_URL", $null, "User")
```

**成本：**
- GPT-4V: $0.01/图片
- 适合测试和小规模使用

#### 选项 C: 使用其他开源视觉模型

- **Yi-VL**: 零一万物开源视觉模型
- **LLaVA**: 开源多模态模型
- **InternVL**: 上海AI实验室开源

需要自行部署或使用云服务。

---

### 方案 3: 混合模式（最佳实践）

结合方案1和方案2：

```python
def analyze_image(self, image_path: str, question: str) -> str:
    try:
        # 尝试使用多模态模型
        response = self.llm.invoke([...])
        return response.content
    except Exception as e:
        # 降级为文本模式
        logger.warning(f"多模态失败，使用文本模式: {e}")
        description = input("请描述图片内容: ")
        return self.chat_text_only(f"图片描述: {description}\n\n问题: {question}")
```

---

## 🎯 推荐方案

### 对于学习/演示目的

**使用方案1（图片描述+文本）**

理由：
- ✅ 立即可用，无需额外配置
- ✅ 理解Agent工作流程
- ✅ 零成本

### 对于生产环境

**使用方案2（真正的多模态模型）**

推荐模型：
1. **通义千问 qwen-vl-max**（性价比高）
2. **OpenAI GPT-4V**（性能最好）
3. **Yi-VL**（开源免费）

---

## 📝 修改代码示例

### 切换到通义千问视觉模型

**步骤1: 获取API Key**

访问 https://dashscope.console.aliyun.com/ 创建API Key

**步骤2: 修改 multimodal_agent.py**

```python
def __init__(self, api_key: Optional[str] = None):
    # 使用通义千问视觉模型
    self.llm = ChatOpenAI(
        model="qwen-vl-max",  # 使用视觉模型
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        openai_api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        temperature=0.3
    )
    self.use_local_model = False
```

**步骤3: 设置环境变量**

```powershell
[System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-xxx", "User")
```

---

## 🧪 测试多模态支持

运行诊断脚本：

```bash
python test_multimodal.py
```

**成功标志：**
```
✅ 成功！
🤖 回答: 这张图片显示了一台笔记本电脑...
```

**失败标志：**
```
❌ 模型返回："抱歉，我没有看到您上传的图片"
```

---

## 📊 模型对比

| 模型 | 多模态 | 成本 | 质量 | 推荐度 |
|------|--------|------|------|--------|
| 当前本地模型 | ❌ | 免费 | - | ⭐ |
| qwen-vl-max | ✅ | ¥0.02/千token | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| GPT-4V | ✅ | $0.01/图片 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Yi-VL | ✅ | 免费 | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| LLaVA | ✅ | 免费 | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 🚀 下一步

1. **立即可用**
   - 使用当前的"图片描述+文本"模式
   - 手动描述图片内容
   - AI提供解决方案

2. **升级多模态**
   - 申请通义千问API Key
   - 修改代码切换模型
   - 享受真正的图片识别

3. **继续开发**
   - 完善错误处理
   - 添加更多功能
   - 优化用户体验

---

## ❓ 常见问题

### Q1: 如何确认本地模型是否支持多模态？

**A:** 查看模型文档或联系部署者确认。常见支持多模态的模型：
- qwen-vl-*
- yi-vl-*
- llava-*
- internvl-*

### Q2: 通义千问视觉模型多少钱？

**A:** 
- qwen-vl-plus: ¥0.008/千token
- qwen-vl-max: ¥0.02/千token
- 新用户有免费额度

### Q3: 可以同时使用多个模型吗？

**A:** 可以！可以实现混合模式：
- 默认使用本地模型（文本）
- 需要图片识别时切换到视觉模型
- 根据需求动态选择

---

## 📞 需要帮助？

如果遇到问题：
1. 查看 `test_multimodal.py` 的诊断输出
2. 检查模型文档
3. 提 Issue 或讨论

---

**当前建议：使用"图片描述+文本"模式进行学习和演示！** 🎯
