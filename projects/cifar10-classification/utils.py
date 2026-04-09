"""
工具函数
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_curves(train_acc_list, test_acc_list, train_loss_list):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 准确率曲线
    ax1.plot(train_acc_list, label='Training Accuracy', linewidth=2)
    ax1.plot(test_acc_list, label='Test Accuracy', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training and Test Accuracy', fontsize=14)
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 损失曲线
    ax2.plot(train_loss_list, label='Training Loss', linewidth=2, color='red')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training Loss', fontsize=14)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("✅ 保存训练曲线: training_curves.png")
    plt.close()


def plot_confusion_matrix(all_labels, all_preds, classes):
    """绘制混淆矩阵"""
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("✅ 保存混淆矩阵: confusion_matrix.png")
    plt.close()


def plot_predictions(test_loader, model, classes, num_samples=16):
    """可视化预测结果"""
    model.eval()
    
    # 获取一批数据
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # 预测
    device = next(model.parameters()).device
    with torch.no_grad():
        images_gpu = images.to(device)
        outputs = model(images_gpu)
        _, predicted = torch.max(outputs, 1)
    
    # 可视化
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * 0.5 + 0.5  # 反标准化
        
        true_label = classes[labels[i]]
        pred_label = classes[predicted[i].item()]
        
        color = 'green' if true_label == pred_label else 'red'
        
        axes[i].imshow(img)
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}',
                         color=color, fontsize=9)
        axes[i].axis('off')
    
    plt.suptitle('Prediction Examples (Green=Correct, Red=Wrong)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    print("✅ 保存预测示例: predictions.png")
    plt.close()
