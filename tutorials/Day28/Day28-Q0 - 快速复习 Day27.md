# Day28-Q0 - 快速复习 Day27

## 🔄 模型部署要点回顾

### 核心流程速记

```
训练模型 → 保存模型 → 构建 API → 容器化 → 云部署
```

**关键技术栈:**
- **模型保存**: PyTorch `state_dict()` / TensorFlow SavedModel
- **Web 框架**: Flask (简单) / FastAPI (高性能)
- **容器化**: Docker + Dockerfile
- **云平台**: Render (免费) / AWS (专业) / GCP (ML友好)

---

## 📝 Day27 知识点检查

### Q1: 为什么要部署
- [ ] 能解释部署的5大原因
- [ ] 知道 API 的作用
- [ ] 理解容器化的价值

### Q2: 模型序列化
- [ ] 会用 `torch.save()` 和 `torch.load()`
- [ ] 知道 `state_dict()` 的优势
- [ ] 了解 ONNX 跨格式转换

### Q3: Flask API
- [ ] 能创建基本的 Flask 应用
- [ ] 会处理文件上传
- [ ] 知道如何添加错误处理

### Q4: FastAPI
- [ ] 理解异步编程的好处
- [ ] 会使用 Pydantic 数据验证
- [ ] 知道依赖注入的概念

### Q5: Docker
- [ ] 会编写 Dockerfile
- [ ] 理解镜像和容器的区别
- [ ] 知道多阶段构建优化

### Q6: 云部署
- [ ] 成功部署到 Render 或其他云平台
- [ ] 会配置环境变量
- [ ] 知道如何查看日志

---

## 💻 代码回顾

### FastAPI 最小应用

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    # 模型推理
    result = model.predict(data)
    return {"prediction": result}
```

### Dockerfile 模板

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 部署命令

```bash
# 构建镜像
docker build -t myapp:latest .

# 运行容器
docker run -p 8000:8000 myapp:latest

# 推送到云端 (Render)
git push origin main  # 自动触发部署
```

---

## 🎯 从 Day27 到 Day28 的过渡

**Day27 我们学会了:**
- ✅ 如何把模型变成 API 服务
- ✅ 如何容器化和部署到云端
- ✅ 如何让全世界使用你的 AI

**Day28 我们要思考:**
- 🤔 AI 会不会有偏见?
- 🤔 如何保护用户隐私?
- 🤔 AI 出错了谁负责?
- 🤔 有哪些法律法规?
- 🤔 AI 从业者有什么责任?

**类比:**
```
Day27: 造出了一辆强大的汽车
   ↓
Day28: 学习交通规则和安全驾驶
```

**技术很重要,但责任更重要!**

---

## 🔗 相关链接

- [← Day27-Q6 - 云平台部署](./Day27-Q6%20-%20云平台部署实战.md)
- [→ Day28-Q1 - AI 偏见和公平性](./Day28-Q1%20-%20AI%20偏见和公平性.md)
