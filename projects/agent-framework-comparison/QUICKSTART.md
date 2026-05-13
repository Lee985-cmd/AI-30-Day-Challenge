# 快速开始指南

## 🚀 5分钟快速体验

### 1. 安装依赖

```bash
cd projects/agent-framework-comparison
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
OPENAI_API_KEY=your-api-key-here
MODEL_NAME=gpt-3.5-turbo
```

**或者使用本地模型**（如Ollama）：

```bash
OPENAI_API_KEY=not-needed
MODEL_NAME=qwen-plus
# 修改代码中的LLM配置指向本地服务
```

### 3. 运行示例

#### 体验LangChain
```bash
python langchain_example.py
```

你会看到：
- ✅ 工具调用Agent演示
- ✅ 对话记忆功能
- ✅ 链式编排示例
- ✅ 自定义工具

#### 体验LlamaIndex
```bash
python llamaindex_example.py
```

你会看到：
- ✅ 文档索引创建
- ✅ 向量检索查询
- ✅ 流式输出
- ✅ 元数据过滤

#### 体验Haystack
```bash
python haystack_example.py
```

你会看到：
- ✅ 文档存储和检索
- ✅ RAG Pipeline
- ✅ 批量处理
- ✅ 性能优化技巧

### 4. 运行性能测试

```bash
python benchmark.py
```

这会：
- 📊 测试三个框架的性能
- 📈 生成对比图表
- 💡 提供分析报告

### 5. 使用框架选择助手

```bash
python framework_selector.py
```

回答7个简单问题，获得个性化推荐！

---

## 💡 常见问题

### Q1: 没有OpenAI API Key怎么办？

**A**: 可以使用本地模型：
- 安装Ollama：https://ollama.ai
- 运行 `ollama pull qwen2.5`
- 修改代码中的LLM配置

或者使用Mock模式（部分示例支持）。

### Q2: 安装依赖时出错？

**A**: 尝试以下方法：
```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或者逐个安装
pip install langchain langchain-openai
pip install llama-index
pip install farm-haystack
```

### Q3: 内存不足？

**A**: Haystack的某些组件比较占用内存，可以：
- 减少文档数量
- 使用更小的嵌入模型
- 关闭不必要的服务

### Q4: 如何只测试一个框架？

**A**: 直接运行对应的示例文件即可：
```bash
python langchain_example.py  # 只测试LangChain
```

---

## 📖 深入学习

1. **阅读配套文章**
   - 查看 `promotion-articles/CSDN/20260513/` 目录
   - 深度理解每个框架的设计理念

2. **研究源代码**
   - 每个示例都有详细注释
   - 尝试修改参数看效果

3. **运行基准测试**
   - 了解各框架的性能特点
   - 根据你的硬件调整预期

4. **使用选择助手**
   - 获取个性化推荐
   - 理解推荐背后的原因

---

## 🎯 下一步

- [ ] 选择一个框架开始你的项目
- [ ] 阅读官方文档深入学习
- [ ] 加入社区获取帮助
- [ ] 分享你的使用体验

**祝你学习愉快！** 🚀
