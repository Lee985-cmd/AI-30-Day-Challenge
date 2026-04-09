"""
模型部署示例 - FastAPI

运行方法:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    
访问文档:
    http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import io
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Model API",
    description="CIFAR-10 图像分类 API",
    version="1.0.0"
)


# ==================== 模型定义 ====================
class SimpleCNN(nn.Module):
    """简单的 CNN 模型（与训练时相同）"""
    
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


# ==================== 全局变量 ====================
model = None
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = datetime.now()
request_count = 0


# ==================== 模型加载 ====================
def load_model():
    """加载训练好的模型"""
    global model
    
    logger.info("📦 加载模型...")
    
    try:
        # 创建模型
        model = SimpleCNN(num_classes=10).to(device)
        
        # 尝试加载权重
        model_path = '../cifar10-classification/cifar_best.pth'
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"✅ 模型加载成功: {model_path}")
        except FileNotFoundError:
            logger.warning("⚠️  未找到预训练模型，使用随机初始化")
        
        model.eval()
        logger.info(f"✅ 模型已就绪 (设备: {device})")
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise


# ==================== 图像处理 ====================
def preprocess_image(image_bytes: bytes):
    """预处理图片"""
    # 打开图片
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 调整大小
    image = image.resize((32, 32))
    
    # 转换为 tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    return image_tensor


# ==================== API 端点 ====================
@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_model()


@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "AI Model API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "uptime_seconds": int(uptime),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    预测接口
    
    上传一张图片，返回分类结果
    """
    global request_count
    
    start_time = time.time()
    
    try:
        # 读取文件
        contents = await file.read()
        
        # 验证文件类型
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="只支持图片文件")
        
        # 预处理
        image_tensor = preprocess_image(contents)
        
        # 推理
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        # 获取所有类别的置信度
        all_predictions = []
        for i, (prob, class_name) in enumerate(zip(probabilities.cpu().numpy(), classes)):
            all_predictions.append({
                "class": class_name,
                "confidence": float(prob)
            })
        
        # 按置信度排序
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        processing_time = (time.time() - start_time) * 1000
        request_count += 1
        
        result = {
            "success": True,
            "prediction": classes[predicted_idx.item()],
            "confidence": float(confidence),
            "all_predictions": all_predictions[:5],  # 返回前5个
            "processing_time_ms": round(processing_time, 2),
            "request_id": request_count
        }
        
        logger.info(f"预测完成: {result['prediction']} ({result['confidence']:.2f})")
        
        return result
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """获取服务指标"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "total_requests": request_count,
        "uptime_seconds": int(uptime),
        "requests_per_second": round(request_count / max(1, uptime), 2),
        "model_device": str(device)
    }


# ==================== 运行 ====================
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
