# ❓ 常见问题 (FAQ)

## 🚀 安装和环境

### Q1: Python 版本要求？

**A:** Python 3.7 或更高版本。推荐使用 Python 3.9。

```bash
# 检查 Python 版本
python --version

# 如果版本太低，从官网下载新版本
# https://www.python.org/downloads/
```

### Q2: 如何创建虚拟环境？

**A:** 
```bash
# 方法 1: 使用 venv (推荐)
python -m venv ai-env
source ai-env/bin/activate  # Linux/Mac
ai-env\Scripts\activate     # Windows

# 方法 2: 使用 conda
conda create -n ai-env python=3.9
conda activate ai-env
```

### Q3: pip install 很慢怎么办？

**A:** 使用国内镜像源：

```bash
# 临时使用
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q4: CUDA/GPU 支持？

**A:** 
```bash
# 1. 检查是否有 NVIDIA GPU
nvidia-smi

# 2. 安装 CUDA 版本的 PyTorch
# 访问 https://pytorch.org/get-started/locally/
# 选择你的 CUDA 版本

# 例如 CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 💻 代码运行

### Q5: ModuleNotFoundError？

**A:** 缺少依赖包。

```bash
# 安装所有依赖
pip install -r requirements.txt

# 或者单独安装缺失的包
pip install numpy
```

### Q6: 数据集下载失败？

**A:** 网络问题，尝试：

```python
# 方法 1: 手动下载
# 访问数据集官网下载，放到指定目录

# 方法 2: 使用代理
import os
os.environ['HTTP_PROXY'] = 'http://your-proxy:port'
os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'

# 方法 3: 使用镜像
# CIFAR-10 国内镜像
# https://mirrors.tuna.tsinghua.edu.cn/help/cifar10/
```

### Q7: CUDA out of memory？

**A:** 显存不足。

```python
# 解决方法 1: 减小 batch size
BATCH_SIZE = 64  # 原来是 128

# 解决方法 2: 清理缓存
import torch
torch.cuda.empty_cache()

# 解决方法 3: 使用 CPU
device = torch.device('cpu')

# 解决方法 4: 梯度累积
accumulation_steps = 4
for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q8: 训练太慢？

**A:** 
```python
# 1. 使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 增加 batch size (如果显存允许)
BATCH_SIZE = 256

# 3. 使用多进程数据加载
DataLoader(..., num_workers=4)

# 4. 混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 📚 学习方法

### Q9: 完全没有编程基础，能学吗？

**A:** 可以！本教程从零开始。

**建议:**
1. 花额外时间学习 Day 1 的 Python 基础
2. 多做练习，不要只看
3. 可以参考额外的 Python 教程：
   - [Python 官方教程](https://docs.python.org/3/tutorial/)
   - [菜鸟教程](https://www.runoob.com/python3/)

### Q10: 数学不好，能学深度学习吗？

**A:** 能！本教程用大白话解释，不强调数学推导。

**重点理解:**
- 直观概念（用比喻）
- 如何使用（代码实现）
- 何时使用（应用场景）

**不需要深入:**
- 复杂的数学证明
- 公式推导细节

### Q11: 每天需要多少时间？

**A:** 建议 2-3 小时/天。

**时间分配:**
- 阅读理论: 30-45 分钟
- 运行代码: 60-90 分钟
- 费曼输出: 30-45 分钟
- 总结反思: 15 分钟

**如果时间不够:**
- 优先运行代码
- 简化费曼输出（写简短笔记）
- 周末补上进度

### Q12: 可以跳过某些天吗？

**A:** 不建议，但可以根据背景调整。

**有编程经验:**
- 可以快速浏览 Day 1
- 重点关注 AI 特定内容

**有 ML 经验:**
- 可以跳过 Week 1
- 直接从 Week 2 开始

**完全零基础:**
- 按顺序学习，不要跳
- 每步都要理解

---

## 🐛 调试和问题

### Q13: 代码报错，怎么办？

**A:** 系统化排查：

```python
# Step 1: 阅读错误信息
# 最后一行通常最重要

# Step 2: 定位错误位置
# 查看 traceback，找到出错的文件和行号

# Step 3: 理解错误类型
# TypeError: 类型错误
# ValueError: 值错误
# AttributeError: 属性错误
# ImportError: 导入错误

# Step 4: 搜索解决方案
# Google: "Python [错误信息]"
# StackOverflow
# GitHub Issues

# Step 5: 打印调试
print(f"变量值: {variable}")
print(f"类型: {type(variable)}")
print(f"形状: {variable.shape}")
```

### Q14: 模型准确率很低？

**A:** 检查清单：

```python
# 1. 数据是否正确？
print(f"数据形状: {X_train.shape}")
print(f"标签范围: {y_train.min()} - {y_train.max()}")

# 2. 数据是否标准化？
# 图像: 像素值应该在 0-1 或 -1 到 1
# 数值特征: 使用 StandardScaler

# 3. 学习率是否合适？
# 太大: Loss 震荡或不下降
# 太小: 训练很慢
# 尝试: 0.1, 0.01, 0.001, 0.0001

# 4. 训练轮数够吗？
# 至少 10-20 epochs

# 5. 模型是否有 bug？
# 检查 forward 函数
# 打印中间层输出形状

# 6. 是否过拟合或欠拟合？
# 过拟合: Train Acc >> Test Acc
# 欠拟合: Train Acc 和 Test Acc 都低
```

### Q15: 如何保存和加载模型？

**A:** 
```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 设置为评估模式
```

---

## 🎯 项目和实践

### Q16: 做完教程后做什么？

**A:** 

**短期 (1-3个月):**
1. 做 2-3 个完整项目
2. 参加 Kaggle 竞赛
3. 写技术博客
4. 贡献开源项目

**中期 (3-6个月):**
1. 选择一个方向深入 (CV/NLP/RL)
2. 阅读论文并复现
3. 建立个人品牌 (GitHub, Blog)
4. 准备求职或深造

**长期 (6-12个月):**
1. 成为某个领域的专家
2. 发表文章或演讲
3. Mentor 其他人
4. 考虑创业或研究

### Q17: 如何构建作品集？

**A:** 

**GitHub 项目应该包含:**
```
project-name/
├── README.md          # 详细的项目说明
├── demo.gif           # 演示动图
├── requirements.txt   # 依赖
├── src/               # 源代码
├── notebooks/         # Jupyter 演示
└── tests/             # 单元测试
```

**README 要点:**
- 项目简介（解决什么问题）
- 技术栈
- 安装和运行方法
- 结果展示（截图/动图）
- 架构说明
- 改进方向

### Q18: 如何选择第一个项目？

**A:** 基于兴趣和能力。

**推荐项目:**
1. **图像分类器** - 识别猫狗、花卉等
2. **情感分析** - 分析电影评论
3. **聊天机器人** - 简单对话系统
4. **目标检测** - 检测特定物体
5. **文本生成** - 生成诗歌或故事

**选择标准:**
- 你有兴趣
- 难度适中（不太简单也不太难）
- 有实际应用场景
- 可以展示给他人

---

## 🌍 社区和资源

### Q19: 遇到问题去哪里提问？

**A:** 

**中文社区:**
- GitHub Issues (本项目)
- 知乎
- CSDN
- V2EX

**国际社区:**
- Stack Overflow
- Reddit (r/MachineLearning, r/learnprogramming)
- PyTorch Forums
- Discord 服务器

**提问技巧:**
```markdown
标题: [简明描述问题]

环境:
- OS: Windows 11
- Python: 3.9
- PyTorch: 2.0.1

问题描述:
[详细说明遇到的问题]

代码:
```python
# 最小可复现代码
```

错误信息:
[完整的错误 traceback]

已尝试的解决方法:
1. ...
2. ...
```

### Q20: 如何保持学习动力？

**A:** 

**设定目标:**
- 每周一个小目标
- 每月一个项目
- 记录进度

**找学习伙伴:**
- 加入学习小组
- 互相督促
- 分享进展

**庆祝小胜利:**
- 完成一天 → 给自己奖励
- 完成一周 → 发朋友圈
- 完成项目 → 写博客

**记住初心:**
- 为什么开始学习 AI？
- 想象学会后的样子
- 看看已经取得的进步

**休息和调整:**
- 累了就休息
- 不要和别人比较
- 每个人节奏不同

---

## 📞 联系和支持

### 还有问题？

- 📧 Email: your.email@example.com
- 💬 Discord: [加入社区](https://discord.gg/invite)
- 🐛 GitHub Issues: [提交问题](https://github.com/Lee985-cmd/AI-30-Day-Challenge/issues)

**我们在这里帮助你！** 💪

---

*最后更新: 2026-04-08*
