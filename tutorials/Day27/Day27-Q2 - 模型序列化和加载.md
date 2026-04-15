# Day27-Q2 - 模型序列化和加载

## 💾 保存你的 AI 大脑

### 问题背景

你花了一周时间训练了一个超牛的图像分类模型,准确率 95%!但现在有个问题:

**每次运行程序都要重新训练?** 
- ❌ 太慢了! (几小时甚至几天)
- ❌ 浪费钱! (GPU 很贵)
- ❌ 结果可能不一样! (随机性)

**解决方案:** 把训练好的模型保存到硬盘,下次直接加载使用!

这就是**模型序列化** (Model Serialization),俗称"保存模型"。

---

## 一、什么是模型序列化?

### 大白话解释

**模型序列化 = 把模型的"记忆"保存到文件里**

就像:
- **训练模型** = 学生学习知识
- **序列化** = 把知识写到笔记本上
- **加载模型** = 下次翻开笔记本复习

### 技术定义

模型序列化是将训练好的机器学习模型的:
1. **参数** (权重和偏置)
2. **结构** (网络架构)
3. **配置** (超参数、优化器状态)

保存到磁盘文件,以便后续加载和使用。

---

## 二、PyTorch 模型保存方法

### 方法1: 只保存参数 (推荐) ⭐

```python
import torch
import torch.nn as nn

# 定义模型
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 创建并训练模型
model = MyModel()
# ... 训练代码 ...

# ✅ 只保存参数 (推荐)
torch.save(model.state_dict(), 'model_weights.pth')

print("模型参数已保存!")
```

**优点:**
- 文件小 (只存参数)
- 灵活 (可以改模型结构)
- 标准做法

**缺点:**
- 需要保留模型类定义
- 加载时要先创建模型对象

### 方法2: 保存整个模型

```python
# 保存整个模型
torch.save(model, 'full_model.pth')

print("完整模型已保存!")
```

**优点:**
- 简单直接
- 不需要模型类定义

**缺点:**
- 文件大
- 依赖具体路径
- 不灵活 (难以修改结构)
- ⚠️ **不推荐用于生产环境**

### 方法3: 保存检查点 (Checkpoint) ⭐⭐⭐

```python
# 保存检查点 (包含更多信息)
checkpoint = {
    'epoch': 50,                    # 训练到第几轮
    'model_state_dict': model.state_dict(),  # 模型参数
    'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态
    'loss': 0.023,                  # 当前损失
    'best_accuracy': 0.95,          # 最佳准确率
    'hyperparameters': {            # 超参数
        'learning_rate': 0.001,
        'batch_size': 64,
    }
}

torch.save(checkpoint, 'checkpoint.pth')

print("检查点已保存!")
```

**优点:**
- 可以中断后继续训练
- 保存了完整的训练状态
- 便于调试和复现

**适用场景:**
- 长时间训练 (可能中断)
- 需要恢复训练
- 实验追踪

---

## 三、加载模型

### 加载方法1: 只加载参数

```python
# 1. 先创建模型结构
model = MyModel()

# 2. 加载参数
model.load_state_dict(torch.load('model_weights.pth'))

# 3. 设置为评估模式 (重要!)
model.eval()

print("模型已加载!")

# 4. 使用模型
with torch.no_grad():  # 推理时不需要梯度
    input_data = torch.randn(1, 10)
    output = model(input_data)
    print(f"预测结果: {output}")
```

**⚠️ 重要提示:**
- `model.eval()` - 切换到评估模式 (关闭 Dropout、BatchNorm 等)
- `torch.no_grad()` - 禁用梯度计算 (节省内存,加快速度)

### 加载方法2: 加载整个模型

```python
# 直接加载
model = torch.load('full_model.pth')
model.eval()

# 使用
with torch.no_grad():
    output = model(input_data)
```

### 加载方法3: 加载检查点

```python
# 加载检查点
checkpoint = torch.load('checkpoint.pth')

# 恢复模型
model = MyModel()
model.load_state_dict(checkpoint['model_state_dict'])

# 恢复优化器 (如果要继续训练)
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 恢复其他信息
start_epoch = checkpoint['epoch']
best_acc = checkpoint['best_accuracy']

print(f"从第 {start_epoch} 轮继续训练")
print(f"历史最佳准确率: {best_acc}")

# 设置为评估模式或训练模式
model.eval()  # 如果只用于推理
# model.train()  # 如果要继续训练
```

---

## 四、不同框架的保存方法

### TensorFlow/Keras

```python
import tensorflow as tf

# 方法1: SavedModel 格式 (推荐)
model.save('my_model')  # 自动创建文件夹

# 加载
loaded_model = tf.keras.models.load_model('my_model')

# 方法2: HDF5 格式
model.save('my_model.h5')

# 加载
loaded_model = tf.keras.models.load_model('my_model.h5')

# 方法3: 只保存权重
model.save_weights('weights.h5')

# 加载 (需要先构建相同结构的模型)
model.load_weights('weights.h5')
```

### Scikit-learn

```python
from sklearn import svm
from sklearn.datasets import make_classification
import joblib

# 训练模型
X, y = make_classification(n_samples=1000, n_features=20)
clf = svm.SVC()
clf.fit(X, y)

# 保存模型
joblib.dump(clf, 'svm_model.pkl')

# 加载模型
loaded_clf = joblib.load('svm_model.pkl')

# 预测
predictions = loaded_clf.predict(X[:5])
```

**注意:** Scikit-learn 用 `joblib` 而不是 `pickle`,因为对 NumPy 数组更高效。

### Hugging Face Transformers

```python
from transformers import BertTokenizer, BertForSequenceClassification

# 保存
model.save_pretrained('./my_bert_model')
tokenizer.save_pretrained('./my_bert_model')

# 加载
model = BertForSequenceClassification.from_pretrained('./my_bert_model')
tokenizer = BertTokenizer.from_pretrained('./my_bert_model')
```

---

## 五、保存格式对比

### 常见格式

| 格式 | 框架 | 优点 | 缺点 |
|------|------|------|------|
| **.pth / .pt** | PyTorch | 原生支持,灵活 | 只能 PyTorch 用 |
| **.h5** | Keras/TensorFlow | 单文件,通用 | 较大 |
| **SavedModel** | TensorFlow | 标准化,含签名 | 文件夹形式 |
| **.pkl** | Scikit-learn/Python | 通用 | 安全性问题 |
| **ONNX** | 跨框架 | 框架无关,可优化 | 转换可能出错 |
| **TensorRT** | NVIDIA GPU | 极致性能 | 只限 NVIDIA |

### ONNX: 跨框架交换格式

**什么是 ONNX?**
- Open Neural Network Exchange
- 让不同框架的模型可以互相转换

**使用示例:**

```python
import torch
import onnx

# 1. PyTorch 导出为 ONNX
model = MyModel()
dummy_input = torch.randn(1, 10)
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    input_names=['input'],
    output_names=['output']
)

# 2. 加载 ONNX 模型 (可以用任何支持 ONNX 的框架)
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")

# 3. 推理
import numpy as np
input_data = np.random.randn(1, 10).astype(np.float32)
outputs = session.run(None, {'input': input_data})
```

**好处:**
- PyTorch 训练的模型可以在 TensorFlow Serving 上部署
- 可以用 TensorRT 优化加速
- 可以在移动端运行

---

## 六、最佳实践

### 实践1: 保存最佳模型

```python
best_accuracy = 0

for epoch in range(100):
    # 训练...
    train_loss = train_one_epoch(model, optimizer, train_loader)
    
    # 验证...
    val_accuracy = evaluate(model, val_loader)
    
    # 如果更好,就保存
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'accuracy': val_accuracy,
        }, 'best_model.pth')
        
        print(f"✅ 保存最佳模型! Acc: {val_accuracy:.4f}")

print(f"训练完成! 最佳准确率: {best_accuracy:.4f}")
```

### 实践2: 定期保存检查点

```python
for epoch in range(100):
    # 训练...
    
    # 每 10 个 epoch 保存一次
    if (epoch + 1) % 10 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'checkpoint_epoch_{epoch}.pth')
        
        print(f"💾 保存检查点: Epoch {epoch}")
```

### 实践3: 版本管理

```python
import os
from datetime import datetime

def save_model_with_version(model, accuracy, base_path='models'):
    """保存带版本信息的模型"""
    
    # 创建版本号 (时间戳 + 准确率)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = f"v{timestamp}_acc{accuracy:.4f}"
    
    # 创建目录
    model_dir = os.path.join(base_path, version)
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
    
    # 保存元数据
    metadata = {
        'version': version,
        'accuracy': accuracy,
        'timestamp': timestamp,
        'architecture': 'MyModel',
        'hyperparameters': {
            'learning_rate': 0.001,
            'batch_size': 64,
        }
    }
    
    import json
    with open(os.path.join(model_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 模型已保存: {version}")
    return version

# 使用
version = save_model_with_version(model, accuracy=0.95)
```

### 实践4: 压缩模型文件

```python
import gzip
import shutil

# 保存后用 gzip 压缩
torch.save(model.state_dict(), 'model.pth')

# 压缩
with open('model.pth', 'rb') as f_in:
    with gzip.open('model.pth.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# 删除原文件
os.remove('model.pth')

print(f"原始大小: {os.path.getsize('model.pth.gz') / 1024:.2f} KB")
```

**压缩率:** 通常可以减少 50-70% 的文件大小

---

## 七、常见问题和坑

### 问题1: CUDA 设备错误

**错误信息:**
```
RuntimeError: Attempting to deserialize object on a CUDA device 
but torch.cuda.is_available() is False.
```

**原因:** 模型在 GPU 上训练,但在 CPU 上加载

**解决:**
```python
# 方法1: 指定加载到 CPU
model.load_state_dict(torch.load('model.pth', map_location='cpu'))

# 方法2: 动态判断
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('model.pth', map_location=device))
```

### 问题2: 模型结构不匹配

**错误信息:**
```
RuntimeError: Error(s) in loading state_dict for MyModel:
    Missing key(s) in state_dict: "fc3.weight".
    Unexpected key(s) in state_dict: "fc1.bias".
```

**原因:** 保存时的模型结构和加载时不一样

**解决:**
- 确保模型类定义一致
- 或者使用整个模型保存方法 (不推荐)

### 问题3: 文件太大

**问题:** 模型文件几个 GB,传输和存储都困难

**解决:**
1. **只保存参数** (不要用 `torch.save(model, ...)`)
2. **量化** (Float32 → Float16 或 Int8)
3. **剪枝** (移除不重要的连接)
4. **压缩** (gzip/zip)

```python
# 半精度量化
model_fp16 = model.half()  # Float32 → Float16
torch.save(model_fp16.state_dict(), 'model_fp16.pth')

# 文件大小减少约 50%
```

### 问题4: 加载后性能差

**问题:** 加载的模型预测结果不对或很慢

**检查清单:**
- [ ] 调用了 `model.eval()` 吗?
- [ ] 使用了 `torch.no_grad()` 吗?
- [ ] 输入数据预处理一致吗?
- [ ] 模型版本对吗?

```python
# ✅ 正确的推理流程
model.load_state_dict(torch.load('model.pth'))
model.eval()  # ← 别忘了!

with torch.no_grad():  # ← 别忘了!
    output = model(input_data)
```

### 问题5: 安全性问题

**警告:** 不要加载不可信的 `.pth` 或 `.pkl` 文件!

**风险:** 这些文件可能包含恶意代码

**安全做法:**
```python
# ❌ 不安全
model = torch.load('untrusted_model.pth')

# ✅ 更安全: 只加载 state_dict
model = MyModel()
model.load_state_dict(torch.load('trusted_model.pth'))
```

---

## 八、实战: 完整的保存和加载流程

### 完整示例

```python
import torch
import torch.nn as nn
import os
import json
from datetime import datetime

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ModelManager:
    """模型管理器 - 处理保存和加载"""
    
    def __init__(self, model_dir='saved_models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, epoch, accuracy, config=None):
        """保存检查点"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}.pth"
        filepath = os.path.join(self.model_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
            'config': config or {},
            'timestamp': timestamp,
        }
        
        torch.save(checkpoint, filepath)
        print(f"✅ 检查点已保存: {filename}")
        
        return filename
    
    def load_checkpoint(self, filepath, model, optimizer=None):
        """加载检查点"""
        
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✅ 检查点已加载: Epoch {checkpoint['epoch']}, "
              f"Acc: {checkpoint['accuracy']:.4f}")
        
        return checkpoint
    
    def export_for_deployment(self, model, input_shape=(1, 3, 224, 224)):
        """导出用于部署的模型"""
        
        # 1. 保存为标准格式
        model_path = os.path.join(self.model_dir, 'deploy_model.pth')
        torch.save(model.state_dict(), model_path)
        
        # 2. 导出为 ONNX
        onnx_path = os.path.join(self.model_dir, 'deploy_model.onnx')
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # 3. 保存元数据
        metadata = {
            'input_shape': list(input_shape),
            'output_classes': 10,
            'framework': 'PyTorch',
            'export_date': datetime.now().isoformat(),
        }
        
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 模型已导出用于部署")
        print(f"   - PyTorch: {model_path}")
        print(f"   - ONNX: {onnx_path}")
        print(f"   - Metadata: {metadata_path}")


# 使用示例
if __name__ == '__main__':
    # 创建模型和管理器
    model = ImageClassifier(num_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    manager = ModelManager('my_models')
    
    # 模拟训练
    print("模拟训练...")
    for epoch in range(5):
        # 训练代码...
        accuracy = 0.85 + epoch * 0.02  # 模拟准确率提升
        
        # 保存检查点
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            accuracy=accuracy,
            config={'lr': 0.001, 'batch_size': 64}
        )
    
    # 导出用于部署
    model.eval()
    manager.export_for_deployment(model)
    
    # 加载模型进行推理
    print("\n加载模型进行推理...")
    new_model = ImageClassifier(num_classes=10)
    
    # 加载最新的检查点
    checkpoints = sorted([
        f for f in os.listdir('my_models') 
        if f.startswith('checkpoint_')
    ])
    
    if checkpoints:
        latest_checkpoint = os.path.join('my_models', checkpoints[-1])
        manager.load_checkpoint(latest_checkpoint, new_model)
        
        # 推理
        new_model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            output = new_model(dummy_input)
            predicted_class = output.argmax(dim=1).item()
            print(f"预测类别: {predicted_class}")
```

---

## 九、本章小结

### 核心要点

✅ **三种保存方法:**
1. **只保存参数** (`state_dict`) - 推荐 ⭐
2. **保存整个模型** - 简单但不灵活
3. **保存检查点** - 适合长时间训练 ⭐⭐⭐

✅ **加载注意事项:**
- 调用 `model.eval()` 切换到评估模式
- 使用 `torch.no_grad()` 禁用梯度
- 处理设备映射 (`map_location`)

✅ **最佳实践:**
- 保存最佳模型 (而非最后一个)
- 定期保存检查点
- 版本管理和元数据
- 压缩和量化减小文件大小

✅ **常见问题:**
- CUDA 设备错误 → 用 `map_location`
- 结构不匹配 → 确保模型定义一致
- 文件太大 → 量化、剪枝、压缩
- 性能差 → 检查 `eval()` 和 `no_grad()`

---

## 🎯 下一步

学会了保存和加载模型,接下来学习如何把它变成 API 服务:

- [Q3](./Day27-Q3%20-%20用%20Flask%20构建%20API.md): 用 Flask 创建简单的预测 API
- [Q4](./Day27-Q4%20-%20用%20FastAPI%20构建高性能%20API.md): 用 FastAPI 提升性能
- [Q5](./Day27-Q5%20-%20Docker%20容器化部署.md): 用 Docker 打包应用

**准备好了吗?让我们开始构建 API!** 🚀

---

## 📱 关于作者 & 获取更多资源

本教程由 **Lee（职场宝爸）** 创建，记录从零基础到独立完成 AI 项目的真实历程。

### 关注公众号，获取独家内容

**公众号名称：Lee 的成长日记**

微信搜索关注，获取：
- ✅ **AI 学习路线规划**：零基础如何系统学习 AI
- ✅ **项目实战源码**：完整可运行的项目代码
- ✅ **深度技术解析**：前沿技术原理 + 手写代码实现
- ✅ **职场成长心得**：一个宝爸的 AI 逆袭之路

**关注福利**：
- 回复「**路线**」→ 获取 30 天 AI 学习计划表
- 回复「**项目**」→ 获取 GitHub 项目源码合集
- 回复「**资料**」→ 获取零基础学习资源推荐

**扫码关注公众号**：

![公众号二维码](../../../images/logos/ewm.jpg)

### 其他平台

- 📂 **GitHub**：https://github.com/Lee985-cmd/AI-30Days-Challenge
- 📝 **CSDN 博客**：https://blog.csdn.net/m0_67081842
- 💬 **公众号**：微信搜索「Lee 的成长日记」

---

> 💡 **学习建议**
> 
> 如果本篇教程对你有帮助，欢迎：
> 1. **Star GitHub 项目**：https://github.com/Lee985-cmd/AI-30Days-Challenge
> 2. **关注公众号**获取更多独家内容
> 3. **留言交流**你的学习困惑
> 
> **一起学习，一起进步！** 🤝
