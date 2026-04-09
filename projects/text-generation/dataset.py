"""
文本数据集
"""

import torch
from torch.utils.data import Dataset
import os


class TextDataset(Dataset):
    """文本数据集"""
    
    def __init__(self, text, seq_length):
        self.seq_length = seq_length
        
        # 创建字符到索引的映射
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        self.vocab_size = len(self.chars)
        
        # 将文本转换为索引序列
        self.data = [self.char_to_idx[ch] for ch in text]
        
        # 创建输入-目标对
        self.n_samples = len(self.data) - seq_length
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return x, y


def load_text_data(file_path='data/poems.txt'):
    """加载文本数据"""
    print(f"\n📖 加载文本数据: {file_path}")
    
    # 如果文件不存在，创建示例数据
    if not os.path.exists(file_path):
        print("⚠️  文件不存在，创建示例数据...")
        os.makedirs('data', exist_ok=True)
        
        sample_text = """床前明月光疑是地上霜
举头望明月低头思故乡
春眠不觉晓处处闻啼鸟
夜来风雨声花落知多少
白日依山尽黄河入海流
欲穷千里目更上一层楼
红豆生南国春来发几枝
愿君多采撷此物最相思
"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"✅ 文本长度: {len(text)} 个字符")
    
    return text


if __name__ == '__main__':
    # 测试
    text = load_text_data()
    dataset = TextDataset(text, seq_length=50)
    
    print(f"词汇表大小: {dataset.vocab_size}")
    print(f"样本数量: {len(dataset)}")
    
    # 测试获取一个样本
    x, y = dataset[0]
    print(f"输入形状: {x.shape}")
    print(f"目标形状: {y.shape}")
