# 数据集说明

## 📁 目录结构

```
datasets/
├── sample_images/      # 示例图片
├── sample_texts/       # 示例文本
└── README.md          # 本文件
```

## 🎯 使用说明

本目录存放小型示例数据集，用于代码演示和测试。

**注意:** 
- ❌ 不要提交大型数据集到 Git
- ✅ 使用 `.gitignore` 忽略大数据文件
- ✅ 提供下载脚本或链接

## 📸 示例图片

### CIFAR-10 样本

位置: `sample_images/cifar10/`

包含 10 个类别的示例图片：
- plane (飞机)
- car (汽车)
- bird (鸟)
- cat (猫)
- deer (鹿)
- dog (狗)
- frog (青蛙)
- horse (马)
- ship (船)
- truck (卡车)

**用途:** 测试图像分类代码

### MNIST 样本

位置: `sample_images/mnist/`

手写数字 0-9 的示例图片

**用途:** 测试基础神经网络

## 📝 示例文本

### 中文诗词

位置: `sample_texts/poetry.txt`

格式: 每行一首诗

```
床前明月光，疑是地上霜。举头望明月，低头思故乡。
春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。
...
```

**用途:** 文本生成项目

### 英文文本

位置: `sample_texts/english.txt`

莎士比亚、简奥斯汀等经典文学作品片段

**用途:** 英文文本生成

### 情感分析数据

位置: `sample_texts/sentiment.csv`

```
text,label
"I love this movie!",positive
"Terrible experience",negative
...
```

**用途:** 情感分析项目

## 📥 获取完整数据集

### CIFAR-10

```python
import torchvision

# 自动下载
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True
)
```

**官网:** https://www.cs.toronto.edu/~kriz/cifar.html  
**大小:** 163 MB

### MNIST

```python
import torchvision

trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True
)
```

**官网:** http://yann.lecun.com/exdb/mnist/  
**大小:** 55 MB

### IMDB 影评

```python
from datasets import load_dataset

dataset = load_dataset('imdb')
```

**HuggingFace:** https://huggingface.co/datasets/imdb  
**大小:** 80 MB

### COCO (目标检测)

**官网:** https://cocodataset.org/  
**大小:** 25 GB (完整) / 1 GB (迷你版)

**下载迷你版:**
```bash
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

### Common Voice (语音)

**官网:** https://commonvoice.mozilla.org/  
**大小:**  varies by language

## 🔗 其他常用数据集

### 计算机视觉

| 数据集 | 任务 | 大小 | 链接 |
|--------|------|------|------|
| ImageNet | 图像分类 | 150 GB | http://image-net.org/ |
| Pascal VOC | 目标检测 | 2 GB | http://host.robots.ox.ac.uk/pascal/VOC/ |
| Cityscapes | 语义分割 | 11 GB | https://www.cityscapes-dataset.com/ |

### 自然语言处理

| 数据集 | 任务 | 大小 | 链接 |
|--------|------|------|------|
| SQuAD | 问答 | 40 MB | https://rajpurkar.github.io/SQuAD-explorer/ |
| GLUE | NLP 基准 | - | https://gluebenchmark.com/ |
| WikiText | 语言模型 | 100 MB | https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/ |

### 音频

| 数据集 | 任务 | 大小 | 链接 |
|--------|------|------|------|
| LibriSpeech | 语音识别 | 60 GB | http://www.openslr.org/12 |
| GTZAN | 音乐分类 | 1.2 GB | http://marsyas.info/downloads/datasets.html |

## 📊 数据集统计

建议在项目中添加数据统计：

```python
import pandas as pd

# 加载数据
df = pd.read_csv('dataset.csv')

# 基本统计
print(f"总样本数: {len(df)}")
print(f"类别分布:\n{df['label'].value_counts()}")
print(f"缺失值:\n{df.isnull().sum()}")

# 可视化
df['label'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.savefig('class_distribution.png')
```

## ⚖️ 数据许可

使用数据集时注意许可证：

- ✅ **允许商用**: CC0, MIT, Apache 2.0
- ⚠️ **需署名**: CC-BY
- ❌ **禁止商用**: CC-BY-NC
- ❌ **禁止修改**: CC-BY-ND

**使用前务必检查许可证！**

## 🛠️ 数据处理工具

### Pandas

```python
import pandas as pd

# 读取 CSV
df = pd.read_csv('data.csv')

# 清洗数据
df = df.dropna()  # 删除缺失值
df = df.drop_duplicates()  # 删除重复

# 保存
df.to_csv('cleaned_data.csv', index=False)
```

### Albumentations (图像增强)

```python
import albumentations as A

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
])

augmented = transform(image=image)
```

### HuggingFace Datasets

```python
from datasets import load_dataset

# 加载任意 HF 数据集
dataset = load_dataset('glue', 'mrpc')

# 预处理
def tokenize_function(examples):
    return tokenizer(examples['sentence1'], examples['sentence2'])

tokenized_datasets = dataset.map(tokenize_function, batched=True)
```

## 💾 存储建议

### 小数据集 (< 100 MB)
- ✅ 可以直接放入 Git
- ✅ 放在 `datasets/` 目录

### 中等数据集 (100 MB - 1 GB)
- ⚠️ 使用 Git LFS
- ⚠️ 或提供下载脚本

### 大数据集 (> 1 GB)
- ❌ 不要放入 Git
- ✅ 提供下载链接
- ✅ 使用云存储 (AWS S3, Google Drive)
- ✅ 添加 `download.sh` 脚本

## 📝 下载脚本示例

```bash
#!/bin/bash
# download_datasets.sh

echo "Downloading datasets..."

# CIFAR-10
echo "Downloading CIFAR-10..."
python -c "import torchvision; torchvision.datasets.CIFAR10(root='./data', download=True)"

# IMDB
echo "Downloading IMDB..."
python -c "from datasets import load_dataset; load_dataset('imdb')"

echo "✅ All datasets downloaded!"
```

使用方法：
```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

## 🔒 隐私和伦理

使用数据集时注意：

1. **个人隐私**
   - 避免使用包含个人身份信息的数据
   - 如需使用，确保已匿名化

2. **偏见问题**
   - 检查数据集是否存在偏见
   - 在文档中说明局限性

3. **版权问题**
   - 尊重知识产权
   - 遵守使用条款

4. **透明度**
   - 说明数据来源
   - 标注采集时间
   - 描述潜在偏差

---

**维护者:** AI 30-Day Challenge Team  
**最后更新:** 2026-04-08  
**贡献欢迎!** 🙏
