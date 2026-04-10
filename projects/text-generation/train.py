"""
训练脚本
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
from model import create_model
from dataset import TextDataset, load_text_data


# 配置参数
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 64
SEQ_LENGTH = 50
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, dataloader, criterion, optimizer, dataset):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_steps = len(dataloader)
    
    # 使用进度条
    progress_bar = tqdm(dataloader, desc=f'Training', leave=False)
    
    for batch_x, batch_y in progress_bar:
        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        
        # 前向传播
        outputs, _ = model(batch_x)
        loss = criterion(outputs.reshape(-1, dataset.vocab_size), batch_y.reshape(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / total_steps
    return avg_loss


def generate_sample(model, dataset, prompt="床前", length=20, temperature=0.8):
    """生成示例文本"""
    model.eval()
    
    input_indices = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
    generated = list(prompt)
    
    with torch.no_grad():
        hidden = model.init_hidden(1, DEVICE)
        input_tensor = torch.tensor([input_indices], dtype=torch.long).to(DEVICE)
        _, hidden = model(input_tensor, hidden)
        
        next_char_idx = input_indices[-1]
        
        for _ in range(length):
            input_tensor = torch.tensor([[next_char_idx]], dtype=torch.long).to(DEVICE)
            output, hidden = model(input_tensor, hidden)
            
            output = output.squeeze() / temperature
            probs = torch.softmax(output, dim=0)
            next_char_idx = torch.multinomial(probs, 1).item()
            
            next_char = dataset.idx_to_char[next_char_idx]
            generated.append(next_char)
    
    return ''.join(generated)


def train(num_epochs=NUM_EPOCHS):
    """主训练函数"""
    print("=" * 60)
    print("文本生成 - 训练")
    print("=" * 60)
    print(f"使用设备: {DEVICE}")
    print(f"训练轮数: {num_epochs}")
    print("=" * 60)
    
    # 1. 加载数据
    text = load_text_data()
    dataset = TextDataset(text, SEQ_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print(f"\n词汇表大小: {dataset.vocab_size}")
    print(f"样本数量: {len(dataset)}")
    
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
    
    # 3. 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 4. 训练
    print("\n🚀 开始训练...")
    print("-" * 60)
    
    loss_history = []
    sample_prompts = ["床前", "春眠", "白日", "红豆"]
    
    for epoch in range(num_epochs):
        avg_loss = train_epoch(model, dataloader, criterion, optimizer, dataset)
        scheduler.step()
        loss_history.append(avg_loss)
        
        # 每 5 个 epoch 打印一次
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
            
            # 生成多个示例
            print("\n✨ 生成示例:")
            for prompt in sample_prompts:
                sample = generate_sample(model, dataset, prompt, length=20, temperature=0.8)
                print(f"  '{prompt}' -> {sample}")
            print("-" * 60)
    
    # 5. 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'model_{timestamp}.pth'
    torch.save(model.state_dict(), model_path)
    torch.save(model.state_dict(), 'model.pth')  # 最新版本
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 6. 保存训练历史
    history = {
        'loss': loss_history,
        'epochs': num_epochs,
        'vocab_size': dataset.vocab_size,
        'timestamp': timestamp
    }
    with open('training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"✅ 训练历史已保存: training_history.json")
    
    # 7. 绘制 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.title('Training Loss Curve', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss 曲线已保存: training_loss.png")
    
    print("\n🎉 训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    train()
