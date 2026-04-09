"""
评估脚本
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import create_model
from utils import plot_confusion_matrix, plot_predictions
from sklearn.metrics import classification_report


# 配置参数
BATCH_SIZE = 128
NUM_CLASSES = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def load_test_data(batch_size=BATCH_SIZE):
    """加载测试集"""
    print("\n📊 加载测试集...")
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"✅ 测试集: {len(test_dataset)} 张图片")
    
    return test_loader


def evaluate_model(model_path='cifar_best.pth'):
    """评估模型"""
    print("=" * 60)
    print("CIFAR-10 模型评估")
    print("=" * 60)
    
    # 1. 加载数据
    test_loader = load_test_data()
    
    # 2. 创建模型
    print("\n🔧 加载模型...")
    model = create_model(num_classes=NUM_CLASSES, device=DEVICE)
    
    # 3. 加载权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"✅ 模型权重加载成功: {model_path}")
    except FileNotFoundError:
        print(f"❌ 模型文件不存在: {model_path}")
        print("💡 提示: 请先运行 train.py 训练模型")
        return
    
    model.eval()
    
    # 4. 评估
    print("\n📊 开始评估...")
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    print(f"\n✅ 测试准确率: {accuracy:.2f}%")
    print(f"✅ 正确预测: {correct}/{total}")
    
    # 5. 分类报告
    print("\n📋 分类报告:")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))
    
    # 6. 可视化
    print("\n📈 生成可视化图表...")
    plot_confusion_matrix(all_labels, all_preds, CLASSES)
    plot_predictions(test_loader, model, CLASSES)
    
    print("\n🎉 评估完成！")
    print("=" * 60)


if __name__ == '__main__':
    evaluate_model()
