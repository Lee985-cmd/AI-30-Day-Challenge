# 🔧 LangChain 新版导入路径修复指南

## 📋 问题说明

LangChain 在最新版本中将模块拆分到多个独立的包中，旧的导入路径已不再适用。

---

## ✅ 修复对照表

### **核心模块导入路径变更**

| 旧导入路径 | 新导入路径 | 所在包 |
|-----------|-----------|--------|
| `from langchain.document_loaders import ...` | `from langchain_community.document_loaders import ...` | `langchain-community` |
| `from langchain.embeddings import ...` | `from langchain_community.embeddings import ...` | `langchain-community` |
| `from langchain.vectorstores import ...` | `from langchain_community.vectorstores import ...` | `langchain-community` |
| `from langchain.chat_models import ...` | `from langchain_community.chat_models import ...` | `langchain-community` |
| `from langchain.prompts import ...` | `from langchain_core.prompts import ...` | `langchain-core` |
| `from langchain.text_splitter import ...` | `from langchain_text_splitters import ...` | `langchain-text-splitters` |
| `from langchain.chains import ...` | `from langchain_classic.chains import ...` | `langchain-classic` |
| `from langchain.memory import ...` | `from langchain_classic.memory import ...` | `langchain-classic` |

---

## 📦 需要安装的包

```bash
pip install langchain-community langchain-core langchain-text-splitters langchain-classic
```

---

## 🔍 本项目具体修改

### **1. knowledge_base.py**

```python
# ❌ 旧代码
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# ✅ 新代码
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
```

---

### **2. intent_agent.py**

```python
# ❌ 旧代码
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ✅ 新代码
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
```

---

### **3. dialogue_agent.py**

```python
# ❌ 旧代码
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ✅ 新代码
from langchain_community.chat_models import ChatTongyi
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
```

---

## 🚀 验证修复

运行测试脚本验证所有导入是否正常：

```bash
cd projects/ai-customer-service
python test_integration.py
```

预期输出：
```
✅ TextLoader 导入成功
✅ ChatTongyi 导入成功
✅ HuggingFaceEmbeddings 导入成功
✅ KnowledgeBase 导入成功
✅ IntentAgent 导入成功
✅ DialogueAgent 导入成功
```

---

## ⚠️ 常见问题

### **Q1: 清除缓存后仍然报错**

**解决方法：**
```bash
# 删除所有 __pycache__ 目录
Get-ChildItem -Path . -Filter __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force

# 删除所有 .pyc 文件
Get-ChildItem -Path . -Filter *.pyc -Recurse | Remove-Item -Force

# 重新安装依赖
pip uninstall langchain langchain-community langchain-text-splitters langchain-classic -y
pip install langchain-community langchain-text-splitters langchain-classic
```

---

### **Q2: 提示找不到某个模块**

**检查是否安装了对应的包：**
```bash
# 检查已安装的包
pip list | findstr langchain

# 应该看到：
# langchain-community
# langchain-core
# langchain-text-splitters
# langchain-classic
```

---

### **Q3: 版本冲突**

**解决方法：**
```bash
# 卸载所有 langchain 相关包
pip uninstall langchain langchain-community langchain-core langchain-text-splitters langchain-classic -y

# 重新安装
pip install langchain-community langchain-core langchain-text-splitters langchain-classic
```

---

## 📚 官方文档参考

- [LangChain 迁移指南](https://python.langchain.com/docs/how_to/installation)
- [langchain-community](https://pypi.org/project/langchain-community/)
- [langchain-core](https://pypi.org/project/langchain-core/)
- [langchain-text-splitters](https://pypi.org/project/langchain-text-splitters/)
- [langchain-classic](https://pypi.org/project/langchain-classic/)

---

## 🎯 最佳实践

### **1. 使用明确的导入**

```python
# ✅ 推荐：明确指定包名
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate

# ❌ 不推荐：使用模糊的导入
from langchain import ...
```

---

### **2. 定期检查依赖**

```bash
# 查看过期的包
pip list --outdated

# 更新特定包
pip install --upgrade langchain-community
```

---

### **3. 使用 requirements.txt 固定版本**

```txt
langchain-community>=0.0.10
langchain-core>=0.1.0
langchain-text-splitters>=0.0.1
langchain-classic>=1.0.0
```

---

## 📝 总结

**关键要点：**
1. ✅ LangChain 已拆分为多个独立包
2. ✅ 使用 `langchain_community.*` 替代 `langchain.*`
3. ✅ 使用 `langchain_core.*` 替代核心功能
4. ✅ 使用 `langchain_classic.*` 替代经典功能
5. ✅ 使用 `langchain_text_splitters.*` 替代文本分割

**记住这个规则：**
- 社区组件 → `langchain_community`
- 核心组件 → `langchain_core`
- 经典组件 → `langchain_classic`
- 文本分割 → `langchain_text_splitters`

---

**祝你使用愉快！** 🎉
