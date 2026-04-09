"""
模型加载和推理
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """CIFAR-10 CNN 模型"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ModelPredictor:
    """模型预测器"""
    
    def __init__(self, model_path='../cifar10-classification/cifar_best.pth', device='cpu'):
        self.device = torch.device(device)
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # 创建模型
        self.model = SimpleCNN(num_classes=10).to(self.device)
        
        # 加载权重
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"✅ 模型加载成功: {model_path}")
        except FileNotFoundError:
            print(f"⚠️  模型文件不存在: {model_path}，使用随机初始化")
            self.model.eval()
    
    def predict(self, image_tensor):
        """预测单张图片"""
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        return {
            'prediction': self.classes[predicted_idx.item()],
            'confidence': float(confidence),
            'all_predictions': [
                {'class': name, 'confidence': float(prob)}
                for prob, name in sorted(
                    zip(probabilities.cpu().numpy(), self.classes),
                    key=lambda x: x[0],
                    reverse=True
                )
            ]
        }


def create_predictor(model_path='../cifar10-classification/cifar_best.pth', device='cpu'):
    """创建预测器"""
    return ModelPredictor(model_path, device)


if __name__ == '__main__':
    # 测试
    predictor = create_predictor()
    print("✅ 预测器创建成功")
