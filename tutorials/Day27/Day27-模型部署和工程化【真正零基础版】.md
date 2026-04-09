# 🚀 Day27: 模型部署和工程化 - 把 AI 用到生产环境【真正零基础版】

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **从 Jupyter Notebook 到线上服务！让 AI 真正创造价值!**  
> **本教程：完整代码 + 详细讲解 + FastAPI 部署实战**

---

## 📚 目录

1. [为什么需要部署？](#为什么需要部署)
2. [模型优化技术](#模型优化技术)
3. [部署方式详解](#部署方式详解)
4. [实战：FastAPI 部署图像分类](#实战 fastapi 部署图像分类)
5. [Docker 容器化](#docker 容器化)
6. [性能监控和维护](#性能监控和维护)

---

## 🤔 为什么需要部署？

### 说人话版本

想象一下这个场景:

```
你在 Jupyter Notebook 里训练好了模型:
准确率 99%! 效果炸裂! 🔥

然后呢？

问题 1: 别人怎么用？
- 总不能让人都来跑你的 notebook 吧？
- 需要封装成 API 或 APP

问题 2: 速度够快吗？
- 预测一次要 5 秒？用户早跑了!
- 需要优化加速

问题 3: 能抗住多少人用？
- 1 个人用还行，1000 个人同时用就崩了？
- 需要考虑并发和扩展

问题 4: 出问题了怎么办？
- 模型预测错了怎么知道？
- 需要监控和日志

这就是部署要解决的问题!
```

**部署的目标:**
- ✅ 让用户能用 (API/APP)
- ✅ 速度快 (优化加速)
- ✅ 稳定可靠 (高可用)
- ✅ 容易维护 (监控日志)

### 从研究到生产的差距

```python
"""
研究环境 (Jupyter Notebook):
✓ 数据是干净的
✓ 代码是一次性的
✓ 不用考虑性能
✓ 单机运行
✗ 无法给别人用
✗ 没有监控
✗ 不可扩展

生产环境 (线上服务):
✓ 7×24 小时服务
✓ 支持高并发
✓ 有监控告警
✓ 自动扩展
✗ 数据可能很脏
✗ 要求高性能
✗ 需要考虑安全

差距怎么弥补？
→ 学习工程化技能!
"""
```

### 真实应用场景

**场景 1: 创业公司**

```
你开发了一个 AI 产品:
- 图像识别判断病虫害

在实验室:
✓ 准确率 95%

上线后:
❌ 农民伯伯不会用 Python
❌ 手机拍照上传要等 10 秒
❌ 100 个人同时用就卡死
❌ 识别错了没人知道

怎么办？
→ 学习部署和优化!
```

**场景 2: 大公司项目**

```
你在互联网公司做推荐系统:

模型团队:
- 训练了超牛的深度学习模型
- 准确率提升 20%

工程团队:
- 这模型跑得太慢了!
- 显存占用太大!
- 没法部署!

怎么办？
→ 模型优化和工程化!
```

---

## 🔧 模型优化技术

### 为什么要优化？

```python
"""
原始模型的问题:

1. 太大
   - ResNet50: 100MB+
   - BERT: 440MB
   - 下载慢，占空间

2. 太慢
   - CPU 上预测一次 1 秒
   - 用户等不起

3. 太耗资源
   - 显存占用 4GB
   - 内存占用 8GB
   - 成本高

优化方法:
1. 量化 (Quantization)
2. 剪枝 (Pruning)
3. 知识蒸馏 (Knowledge Distillation)
4. 使用更高效的架构
"""
```

### 优化方法 1: 量化 (Quantization)

```python
"""
量化是什么？

说人话:
把高精度的数字变成低精度

FP32(单精度浮点数) → INT8(8 位整数)
32 位 → 8 位
4 倍压缩!

好处:
✓ 模型变小 (4 倍)
✓ 速度变快 (2-4 倍)
✓ 显存占用减少
✓ 几乎不影响准确率

代价:
✗ 精度损失 0.5-2%
✗ 某些操作不支持

就像照片压缩:
原图 10MB → JPG 2MB
肉眼看不出区别
"""

# PyTorch 量化示例
import torch
import torchvision.models as models

print("=" * 60)
print("模型量化演示")
print("=" * 60)

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

print(f"\n原始模型:")
print(f"  - 参数量：{sum(p.numel() for p in model.parameters()):,}")
print(f"  - 数据类型：FP32 (32 位浮点)")

# 动态量化 (最简单)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # 量化哪些层
    dtype=torch.qint8   # 量化成什么类型
)

print(f"\n量化后模型:")
print(f"  - 大小：约原来的 1/4")
print(f"  - 速度：约原来的 2-3 倍")
print(f"  - 精度损失：< 1%")

# 保存量化模型
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
print("\n✓ 量化模型已保存")

"""
量化类型:

1. 训练后量化 (Post-training Quantization)
   - 训练好的模型直接量化
   - 简单快速
   - 精度损失稍大

2. 量化感知训练 (Quantization-Aware Training)
   - 训练时就模拟量化
   - 模型适应低精度
   - 精度损失小

3. 动态量化 (Dynamic Quantization)
   - 推理时动态量化
   - 最灵活
   - 适合 NLP 模型
"""
```

### 优化方法 2: 剪枝 (Pruning)

```python
"""
剪枝是什么？

说人话:
把不重要的神经元剪掉

就像修剪树枝:
- 剪掉枯枝败叶
- 树长得更好
- 还不影响开花结果

原理:
- 有些权重接近 0，没啥用
- 把这些权重设为 0
- 模型变小变快

类型:
1. 非结构化剪枝
   - 随机剪掉单个权重
   - 压缩率高
   - 需要特殊硬件支持

2. 结构化剪枝
   - 剪掉整个通道/滤波器
   - 普通硬件也能加速
   - 压缩率低一些
"""

# PyTorch 剪枝示例
import torch.nn.utils.prune as prune

print("\n" + "=" * 60)
print("模型剪枝演示")
print("=" * 60)

# 创建简单模型
model = torch.nn.Linear(10, 10)

print(f"\n原始权重:")
print(model.weight)

# L1 范数剪枝 (剪掉最小的 30%)
prune.l1_unstructured(
    module=model,
    name='weight',
    amount=0.3  # 剪掉 30%
)

print(f"\n剪枝后权重:")
print(model.weight)
print(f"\n零值数量：{(model.weight == 0).sum().item()} / {model.weight.numel()}")

"""
剪枝建议:

CNN 模型:
- 卷积层：剪掉 20-40%
- 全连接层：剪掉 50-80%

Transformer:
- Attention 层：小心剪 (10-20%)
- FFN 层：可以多剪 (40-60%)

注意:
- 剪完要 fine-tune
- 逐步剪，不要一次剪太多
- 验证集监控准确率
"""
```

### 优化方法 3: 知识蒸馏 (Knowledge Distillation)

```python
"""
知识蒸馏是什么？

比喻:
教授 (大模型) → 教 → 学生 (小模型)

教授知识渊博但行动慢
学生学习快而且能考高分

过程:
1. 用大模型 (Teacher) 做预测
2. 让小模型 (Student) 学习 Teacher 的输出
3. Student 变得又小又快又好

优势:
✓ 模型很小
✓ 速度很快
✓ 效果接近大模型

应用:
- TinyBERT (蒸馏自 BERT)
- MobileNet (手机端 CNN)
- DistilGPT (轻量级 GPT)
"""

# 知识蒸馏伪代码
def knowledge_distillation(teacher_model, student_model, 
                          data, temperature=4.0):
    """
    知识蒸馏简化版
    
    temperature: 温度参数
    - 越高，输出越平滑
    - 学生学得越好
    """
    
    for batch in data:
        # Teacher 的预测 (软标签)
        with torch.no_grad():
            teacher_logits = teacher_model(batch)
            teacher_probs = softmax(teacher_logits / temperature)
        
        # Student 的预测
        student_logits = student_model(batch)
        student_probs = softmax(student_logits / temperature)
        
        # 蒸馏损失 (让学生模仿老师)
        distill_loss = KL_divergence(
            student_probs, 
            teacher_probs
        )
        
        # 反向传播更新学生
        optimizer.zero_grad()
        distill_loss.backward()
        optimizer.step()

"""
蒸馏技巧:

1. Temperature 选择
   - 通常用 3-5
   - 太高会模糊
   - 太低学不到东西

2. 损失函数
   - 蒸馏损失 (模仿老师)
   - 真实损失 (正确答案)
   - 加权组合

3. 架构设计
   - Student 要足够小
   - 但要有足够的容量
   - MobileNetV2/V3 是好选择
"""
```

---

## 📦 部署方式详解

### 部署方式对比

```python
"""
常见部署方式:

1. ONNX (Open Neural Network Exchange)
   - 微软、Facebook 等搞的开放格式
   - 不同框架之间转换
   - PyTorch → ONNX → TensorRT
   
   优点:
   ✓ 跨框架
   ✓ 加速明显
   ✓ 工具链成熟
   
   缺点:
   ✗ 某些操作不支持
   ✗ 调试困难


2. TensorRT (NVIDIA GPU 加速)
   - NVIDIA 亲儿子
   - 专门优化 GPU 推理
   
   优点:
   ✓ 速度飞快 (10 倍+)
   ✓ 显存占用少
   ✓ 支持 FP16/INT8
   
   缺点:
   ✗ 只能用 NVIDIA GPU
   ✗ 配置复杂


3. OpenVINO (Intel CPU/VPU 加速)
   - Intel 的方案
   - 主打 CPU 推理
   
   优点:
   ✓ CPU 速度快
   ✓ 支持 Intel 各种硬件
   ✓ 免费开源
   
   缺点:
   ✗ GPU 支持一般
   ✗ 主要支持 Intel 硬件


4. TorchScript (PyTorch 原生)
   - PyTorch 自带的序列化
   - 最简单
   
   优点:
   ✓ 无需转换
   ✓ 完美兼容
   ✓ 支持动态图
   
   缺点:
   ✗ 加速有限
   ✗ 只能 PyTorch 生态
"""
```

### ONNX 转换实战

```python
"""
ONNX 转换流程:

PyTorch 模型 → 导出 → ONNX 格式 → 推理引擎加载 → 加速推理
"""

import torch
import torchvision.models as models

print("=" * 60)
print("ONNX 模型转换演示")
print("=" * 60)

# 加载模型
model = models.resnet18(pretrained=True)
model.eval()

# 假输入 (用于追踪计算图)
dummy_input = torch.randn(1, 3, 224, 224)

# 导出为 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",          # 输出文件名
    export_params=True,        # 保存参数
    opset_version=11,          # ONNX 版本
    do_constant_folding=True,  # 常量折叠优化
    input_names=['input'],     # 输入名
    output_names=['output'],   # 输出名
    dynamic_axes={             # 动态轴 (支持不同 batch size)
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print("✓ 模型已导出为 'resnet18.onnx'")
print("\nONNX 模型优势:")
print("  - 可在多种设备上运行")
print("  - 用 ONNX Runtime 加速")
print("  - 支持 CPU/GPU/FPGA 等")

# 验证模型
import onnx

# 检查模型是否正确
onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)
print("\n✓ ONNX 模型验证通过!")

# 查看模型信息
print(f"\n模型信息:")
print(f"  - 输入：{onnx_model.graph.input[0].name}")
print(f"  - 输出：{onnx_model.graph.output[0].name}")
print(f"  - 节点数：{len(onnx_model.graph.node)}")

"""
使用 ONNX Runtime 推理:

安装:
pip install onnxruntime

使用:
import onnxruntime as ort

session = ort.InferenceSession("resnet18.onnx")
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image})

加速效果:
- CPU: 2-3 倍
- GPU: 5-10 倍
"""
```

---

## 🌐 实战：FastAPI 部署图像分类

让我们实现一个完整的图像分类 API:

```python
# ============================================================================
# 第一部分：导入库
# ============================================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import uvicorn
import time
from datetime import datetime
import logging

print("=" * 60)
print("FastAPI 图像分类服务 - 从零开始")
print("=" * 60)

# ============================================================================
# 第二部分：配置日志
# ============================================================================

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 第三部分：加载模型
# ============================================================================

print("\n正在加载模型...")

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 如果有 GPU 就用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"✓ 模型加载成功!")
print(f"  - 模型：ResNet-18")
print(f"  - 设备：{device}")

# ImageNet 类别标签 (简化版，只列一部分)
IMAGENET_CLASSES = {
    0: 'tench', 1: 'goldfish', 2: 'great white shark',
    3: 'tiger shark', 4: 'hammerhead', 5: 'electric ray',
    # ... 实际有 1000 类
    281: 'tabby cat', 282: 'Persian cat',
    283: 'Siamese cat', 284: 'Egyptian cat',
    285: 'cougar',  # ... 更多类别
}

# 为了方便演示，我们只用几个类别
# 实际应用需要完整的 1000 类映射

# 图片预处理
transform = transforms.Compose([
    transforms.Resize(256),           # 缩放到 256
    transforms.CenterCrop(224),       # 中心裁剪到 224
    transforms.ToTensor(),            # 转 Tensor
    transforms.Normalize(             # 归一化
        mean=[0.485, 0.456, 0.406],  # ImageNet 均值
        std=[0.229, 0.224, 0.225]    # ImageNet 标准差
    )
])

print("✓ 预处理流程配置完成")

# ============================================================================
# 第四部分：创建 FastAPI 应用
# ============================================================================

app = FastAPI(
    title="图像分类 API",
    description="基于 ResNet-18 的图像识别服务",
    version="1.0.0"
)

# 健康检查接口
@app.get("/")
async def root():
    """根路径 - 健康检查"""
    return {
        "message": "图像分类服务运行中",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

# 预测接口
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    图像分类预测接口
    
    参数:
    file: 上传的图片文件
    
    返回:
    JSON 格式的分类结果
    """
    
    start_time = time.time()
    
    # 1. 读取图片
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        logger.error(f"图片读取失败：{e}")
        raise HTTPException(status_code=400, detail="图片格式错误")
    
    # 2. 预处理
    try:
        input_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"预处理失败：{e}")
        raise HTTPException(status_code=400, detail="图片处理失败")
    
    # 3. 推理
    try:
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            top_prob, top_class = torch.topk(probabilities, k=5)
    except Exception as e:
        logger.error(f"推理失败：{e}")
        raise HTTPException(status_code=500, detail="模型推理失败")
    
    # 4. 构建结果
    results = []
    for prob, class_id in zip(top_prob, top_class):
        class_id = class_id.item()
        class_name = IMAGENET_CLASSES.get(class_id, f"class_{class_id}")
        results.append({
            "class_id": class_id,
            "class_name": class_name,
            "probability": round(prob.item(), 4)
        })
    
    # 5. 记录耗时
    inference_time = time.time() - start_time
    
    logger.info(f"预测完成：{results[0]['class_name']} "
                f"(耗时：{inference_time:.3f}s)")
    
    return {
        "predictions": results,
        "inference_time": f"{inference_time:.3f}s",
        "image_size": image.size
    }

# 批量预测接口
@app.post("/predict/batch")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    批量预测接口
    
    参数:
    files: 多张图片
    
    返回:
    所有图片的分类结果
    """
    
    results = []
    
    for file in files:
        # 复用单个预测的逻辑
        result = await predict(file)
        results.append({
            "filename": file.filename,
            "prediction": result
        })
    
    return {"results": results}

# ============================================================================
# 第五部分：添加性能监控
# ============================================================================

from prometheus_fastapi_instrumentator import Instrumentator

# 添加性能监控中间件
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

print("✓ 性能监控已启用")
print("  - 指标：/metrics")
print("  - 包含：请求数、延迟、错误率等")

# ============================================================================
# 第六部分：添加缓存机制
# ============================================================================

from functools import lru_cache
import hashlib

# 简单的结果缓存
cache = {}
CACHE_TTL = 300  # 缓存 5 分钟

def get_image_hash(image_bytes):
    """计算图片哈希值 (用于缓存)"""
    return hashlib.md5(image_bytes).hexdigest()

@app.post("/predict/cached")
async def predict_with_cache(file: UploadFile = File(...)):
    """带缓存的预测接口"""
    
    # 读取图片
    contents = await file.read()
    
    # 计算哈希
    img_hash = get_image_hash(contents)
    
    # 检查缓存
    if img_hash in cache:
        cached_result, timestamp = cache[img_hash]
        
        # 检查是否过期
        if time.time() - timestamp < CACHE_TTL:
            logger.info(f"使用缓存结果：{img_hash}")
            cached_result["from_cache"] = True
            return cached_result
    
    # 重新预测
    from io import BytesIO
    file_io = BytesIO(contents)
    file_io.name = file.filename
    
    # 创建新的 UploadFile 对象
    new_file = UploadFile(file=file_io, filename=file.filename)
    result = await predict(new_file)
    
    # 存入缓存
    cache[img_hash] = (result, time.time())
    
    result["from_cache"] = False
    return result

print("✓ 缓存机制已启用")
print(f"  - 缓存时间：{CACHE_TTL}秒")
print(f"  - 缓存键：MD5 哈希")

# ============================================================================
# 第七部分：错误处理和限流
# ============================================================================

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# 限流器
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# 限流装饰器
@app.post("/predict/limited")
@limiter.limit("10/minute")  # 每分钟最多 10 次
async def limited_predict(request, file: UploadFile = File(...)):
    """限流的预测接口"""
    return await predict(file)

print("✓ 限流已启用")
print("  - 限制：10 次/分钟")
print("  - 防止滥用")

# ============================================================================
# 第八部分：启动服务器
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("启动图像分类服务!")
    print("=" * 60)
    
    print("""
服务信息:
  - 地址：http://127.0.0.1:8000
  - 文档：http://127.0.0.1:8000/docs
  - 健康检查：GET /
  - 预测接口：POST /predict
  - 批量预测：POST /predict/batch
  - 性能指标：GET /metrics

测试方法:

1. 浏览器访问文档:
   http://127.0.0.1:8000/docs

2. 用 curl 测试:
   curl -X POST "http://127.0.0.1:8000/predict" \\
        -F "file=@test_image.jpg"

3. 用 Python 测试:
   ```python
   import requests
   
   files = {'file': open('test.jpg', 'rb')}
   response = requests.post('http://127.0.0.1:8000/predict', files=files)
   print(response.json())
   ```

按 Ctrl+C 停止服务
""")
    
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

# ============================================================================
# 第九部分：Docker 容器化部署
# ============================================================================

"""
Dockerfile 示例:

FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 下载模型 (可以预先下载好)
RUN python download_model.py

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

构建和运行:

# 构建镜像
docker build -t image-classifier:latest .

# 运行容器
docker run -d -p 8000:8000 --name classifier image-classifier:latest

# 查看日志
docker logs -f classifier

# 停止服务
docker stop classifier
"""

print("\n" + "=" * 60)
print("Docker 部署指南")
print("=" * 60)

docker_guide = """
Docker 部署步骤:

1. 创建 Dockerfile (见上面代码)

2. 创建 requirements.txt:
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
torch==2.1.0
torchvision==0.16.0
Pillow==10.1.0
prometheus-fastapi-instrumentator==6.1.0
slowapi==0.1.7

3. 构建镜像:
docker build -t image-classifier:v1 .

4. 运行容器:
docker run -d -p 8000:8000 image-classifier:v1

5. 测试:
curl http://localhost:8000/

优势:
✓ 环境一致 (开发=生产)
✓ 一键部署
✓ 容易扩展
✓ 资源隔离

生产环境建议:
- 用 docker-compose 编排
- 用 Kubernetes 管理
- 配置健康检查
- 设置资源限制
"""

print(docker_guide)

# ============================================================================
# 第十部分：性能优化建议
# ============================================================================

print("\n" + "=" * 60)
print("性能优化建议")
print("=" * 60)

optimization_tips = """
【模型层面】

1. 模型量化
   - FP32 → INT8
   - 速度提升 2-4 倍
   - 精度损失 < 1%

2. 模型剪枝
   - 剪掉不重要的连接
   - 减少计算量
   - 需要 fine-tune

3. 知识蒸馏
   - 大模型教小模型
   - 保持效果
   - 大幅加速

4. 选择高效架构
   - MobileNet (移动端)
   - EfficientNet (平衡)
   - ShuffleNet (超低延迟)


【服务层面】

1. 批处理
   - 合并多个请求
   - 充分利用 GPU
   - 降低平均延迟

2. 异步处理
   - FastAPI 的 async/await
   - 非阻塞 I/O
   - 提高并发

3. 缓存
   - 热门结果缓存
   - 减少重复计算
   - Redis/Memcached

4. 负载均衡
   - 多实例部署
   - Nginx/HAProxy
   - 水平扩展


【硬件层面】

1. GPU 加速
   - NVIDIA TensorRT
   - CUDA 优化
   - 混合精度

2. 专用芯片
   - TPU (Google)
   - NPU (华为)
   - FPGA

3. 边缘计算
   - 就近处理
   - 降低延迟
   - 节省带宽


【监控层面】

1. 性能监控
   - Prometheus + Grafana
   - 延迟、吞吐量
   - 错误率

2. 日志系统
   - ELK Stack
   - 结构化日志
   - 实时告警

3. 链路追踪
   - Jaeger
   - Zipkin
   - 定位瓶颈
"""

print(optimization_tips)

# ============================================================================
# 第十一部分：完整项目结构
# ============================================================================

print("\n" + "=" * 60)
print("推荐的项目结构")
print("=" * 60)

project_structure = """
project/
├── app/                      # 主应用目录
│   ├── __init__.py
│   ├── main.py              # FastAPI 入口
│   ├── api/                 # API 路由
│   │   ├── __init__.py
│   │   ├── predict.py       # 预测接口
│   │   └── health.py        # 健康检查
│   ├── models/              # 模型相关
│   │   ├── __init__.py
│   │   ├── resnet.py        # 模型定义
│   │   └── loader.py        # 模型加载
│   ├── services/            # 业务逻辑
│   │   ├── __init__.py
│   │   ├── predictor.py     # 预测服务
│   │   └── cache.py         # 缓存服务
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   ├── preprocess.py    # 预处理
│   │   └── metrics.py       # 监控指标
│   └── config.py            # 配置文件
├── tests/                   # 测试目录
│   ├── __init__.py
│   ├── test_api.py          # API 测试
│   └── test_model.py        # 模型测试
├── models/                  # 模型文件
│   └── resnet18.pth
├── data/                    # 数据目录
│   ├── train/
│   └── test/
├── scripts/                 # 脚本目录
│   ├── train.py
│   ├── export_onnx.py
│   └── download_model.py
├── docker/                  # Docker 配置
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/                    # 文档目录
│   ├── api.md
│   └── deployment.md
├── requirements.txt         # Python 依赖
├── setup.py                 # 安装包配置
├── README.md                # 项目说明
└── .env                     # 环境变量
"""

print(project_structure)

print("\n🎉 恭喜你完成了模型部署实战!")
print("\n下一步:")
print("  1. 学习 Docker 和 Kubernetes")
print("  2. 了解 CI/CD 流程")
print("  3. 掌握监控和告警")
print("  4. 实践微服务架构")

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day26](../Day26/README.md)
- [→ Day28](../Day28/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*
