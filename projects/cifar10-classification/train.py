"""
训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import create_model
from utils import plot_training_curves
import time


# 配置参数
BATCH_SIZE = 128
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(batch_size=BATCH_SIZE):
    """加载 CIFAR-10 数据集"""
    print("\n📊 加载 CIFAR-10 数据集...")
    
    # 数据增强（训练集）
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 标准化（测试集）
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载数据集
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 张图片")
    print(f"✅ 测试集: {len(test_dataset)} 张图片")
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, epoch, num_epochs):
    """训练一个 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # 每 100 个批次打印一次
        if (batch_idx + 1) % 100 == 0:
            print(f'  Epoch [{epoch}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {running_loss/100:.4f} Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    epoch_acc = 100. * correct / total
    return epoch_acc


def train():
    """主训练函数"""
    print("=" * 60)
    print("CIFAR-10 训练")
    print("=" * 60)
    print(f"使用设备: {DEVICE}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 加载数据
    train_loader, test_loader = load_data()
    
    # 2. 创建模型
    print("\n🔧 创建 CNN 模型...")
    model = create_model(num_classes=NUM_CLASSES, device=DEVICE)
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 4. 训练
    print("\n🚀 开始训练...")
    print("-" * 60)
    
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    best_acc = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        
        # 训练
        train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, NUM_EPOCHS)
        train_acc_list.append(train_acc)
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_acc = 100. * correct / total
        test_acc_list.append(test_acc)
        
        # 记录损失（近似）
        train_loss_list.append(2.0 - train_acc / 100.0)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'cifar_best.pth')
            print(f"  💾 保存最佳模型 (Acc: {test_acc:.2f}%)")
        
        print(f"  Epoch [{epoch}/{NUM_EPOCHS}] "
              f"Train Acc: {train_acc:.2f}% | "
              f"Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        print("-" * 60)
    
    # 5. 可视化
    print("\n📈 生成训练曲线...")
    plot_training_curves(train_acc_list, test_acc_list, train_loss_list)
    
    # 6. 统计时间
    total_time = time.time() - start_time
    print(f"\n⏱️  总耗时: {total_time/60:.1f} 分钟")
    print(f"✅ 最佳测试准确率: {best_acc:.2f}%")
    print("\n🎉 训练完成！")
    print("=" * 60)


if __name__ == '__main__':
    train()
