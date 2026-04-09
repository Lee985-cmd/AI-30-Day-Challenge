"""
文本生成脚本
"""

import torch
from model import create_model
from dataset import TextDataset, load_text_data


# 配置参数
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
SEQ_LENGTH = 50
TEMPERATURE = 0.8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_text(model, dataset, prompt, length=50, temperature=TEMPERATURE):
    """生成文本"""
    model.eval()
    
    # 将提示文本转换为索引
    input_indices = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
    generated = list(prompt)
    
    with torch.no_grad():
        # 初始隐藏状态
        hidden = model.init_hidden(1, DEVICE)
        
        # 处理提示文本
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(DEVICE)
        _, hidden = model(input_tensor, hidden)
        
        # 获取最后一个字符的索引
        next_char_idx = input_indices[-1]
        
        # 生成新字符
        for _ in range(length):
            # 准备输入
            input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(DEVICE)
            
            # 前向传播
            output, hidden = model(input_tensor, hidden)
            
            # 应用温度
            output = output.squeeze() / temperature
            probs = torch.softmax(output, dim=0)
            
            # 采样
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # 添加到生成的文本
            next_char = dataset.idx_to_char[next_char_idx]
            generated.append(next_char)
    
    return ''.join(generated)


def generate(prompt="床前", length=50, temperature=TEMPERATURE, model_path='model.pth'):
    """主生成函数"""
    print("=" * 60)
    print("文本生成")
    print("=" * 60)
    print(f"提示: {prompt}")
    print(f"长度: {length}")
    print(f"温度: {temperature}")
    print("=" * 60)
    
    # 1. 加载数据（获取词汇表）
    text = load_text_data()
    dataset = TextDataset(text, SEQ_LENGTH)
    
    # 2. 创建模型
    print("\n🔧 创建模型...")
    model = create_model(
        vocab_size=dataset.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        device=DEVICE
    )
    
    # 3. 加载权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ 模型加载成功: {model_path}")
    except FileNotFoundError:
        print(f"❌ 模型文件不存在: {model_path}")
        print("💡 提示: 请先运行 train.py 训练模型")
        return
    
    # 4. 生成文本
    print(f"\n✨ 生成文本:")
    print("-" * 60)
    
    generated = generate_text(model, dataset, prompt, length, temperature)
    print(generated)
    print("-" * 60)
    
    # 5. 保存到文件
    with open('generated_samples.txt', 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Prompt: {prompt}, Temperature: {temperature}\n")
        f.write(generated + "\n")
    
    print("✅ 已保存到: generated_samples.txt")
    print("\n🎉 生成完成！")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='文本生成')
    parser.add_argument('--prompt', type=str, default='床前', help='提示文本')
    parser.add_argument('--length', type=int, default=50, help='生成长度')
    parser.add_argument('--temperature', type=float, default=TEMPERATURE, help='温度参数')
    
    args = parser.parse_args()
    
    generate(args.prompt, args.length, args.temperature)
