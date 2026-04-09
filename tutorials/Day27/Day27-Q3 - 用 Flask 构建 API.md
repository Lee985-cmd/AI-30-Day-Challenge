# Day27-Q3 - 用 Flask 构建 API

## 🌐 把你的模型变成 Web 服务

### 问题背景

你已经训练好了模型,也学会了保存和加载。但现在的问题是:

**怎么让别人通过网络调用你的模型?**

- ❌ 把代码发给每个人? (太麻烦!)
- ❌ 让他们安装 Python 和依赖? (不现实!)
- ✅ 提供一个网址,他们发送 HTTP 请求就能得到预测结果! (完美!)

这就是 **API (Application Programming Interface)** - 应用程序接口。

---

## 一、什么是 API?

### 大白话解释

**API = 餐厅的服务员**

你去餐厅吃饭:
- **你** = 客户端 (Client)
- **服务员** = API
- **厨房** = 服务器/模型
- **菜单** = API 文档

**流程:**
```
你点菜 (发送请求) 
  → 服务员接单 (API 接收) 
  → 厨房做菜 (模型推理) 
  → 服务员上菜 (返回结果)
```

### 技术定义

API 是一组规则和协议,允许不同的软件系统相互通信。

**Web API 常见类型:**
- **REST API** - 最常用,基于 HTTP
- **GraphQL** - 灵活查询
- **gRPC** - 高性能,适合微服务

我们今天学 **REST API**,因为它最简单、最通用。

---

## 二、Flask 基础

### 什么是 Flask?

Flask 是一个轻量级的 Python Web 框架,特点是:
- ✅ 简单易学
- ✅ 灵活自由
- ✅ 适合小项目和原型
- ❌ 性能一般 (但对于大多数场景够用)

### 安装 Flask

```bash
pip install flask
```

### Hello World 示例

```python
from flask import Flask

# 创建 Flask 应用
app = Flask(__name__)

# 定义路由 (URL 路径)
@app.route('/')
def hello():
    return 'Hello, World!'

# 运行服务器
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**运行后访问:** `http://localhost:5000`

你会看到: `Hello, World!`

---

## 三、构建预测 API

### 场景: 图像分类服务

假设你有一个训练好的图像分类模型,现在要把它变成 API。

### 步骤1: 准备模型

```python
# model.py
import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        return self.network(x)

# 加载模型
def load_model(model_path='model.pth'):
    model = ImageClassifier(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()  # 设置为评估模式
    return model

# 类别标签
CLASS_NAMES = ['猫', '狗', '鸟', '鱼', '兔子']
```

### 步骤2: 创建 Flask API

```python
# app.py
from flask import Flask, request, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from model import load_model, CLASS_NAMES

# 创建 Flask 应用
app = Flask(__name__)

# 加载模型 (全局变量,只加载一次)
model = load_model('model.pth')

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 定义预测接口
@app.route('/predict', methods=['POST'])
def predict():
    """
    接收上传的图片,返回分类结果
    
    请求格式: multipart/form-data (上传图片文件)
    响应格式: JSON
    """
    
    # 1. 检查是否有文件
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    
    try:
        # 2. 读取图片
        image = Image.open(file.stream).convert('RGB')
        
        # 3. 预处理
        input_tensor = transform(image).unsqueeze(0)  # 添加 batch 维度
        
        # 4. 推理
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 5. 构建响应
        result = {
            'success': True,
            'prediction': CLASS_NAMES[predicted_class],
            'class_id': predicted_class,
            'confidence': round(confidence, 4),
            'all_probabilities': {
                CLASS_NAMES[i]: round(prob.item(), 4) 
                for i, prob in enumerate(probabilities[0])
            }
        }
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# 健康检查接口
@app.route('/health', methods=['GET'])
def health_check():
    """检查服务是否正常运行"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True
    }), 200


# 运行服务器
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 步骤3: 运行和测试

**启动服务器:**
```bash
python app.py
```

**测试方法1: 使用 curl**
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@cat.jpg"
```

**测试方法2: 使用 Python requests**
```python
import requests

# 发送图片
with open('cat.jpg', 'rb') as f:
    files = {'file': ('cat.jpg', f, 'image/jpeg')}
    response = requests.post('http://localhost:5000/predict', files=files)

# 查看结果
print(response.json())
# {
#   "success": true,
#   "prediction": "猫",
#   "class_id": 0,
#   "confidence": 0.9234,
#   "all_probabilities": {
#     "猫": 0.9234,
#     "狗": 0.0456,
#     "鸟": 0.0123,
#     "鱼": 0.0098,
#     "兔子": 0.0089
#   }
# }
```

**测试方法3: 使用 Postman**
1. 打开 Postman
2. 新建 POST 请求: `http://localhost:5000/predict`
3. Body → form-data
4. Key: `file`, Type: File, 选择图片
5. 点击 Send

---

## 四、文本分类 API

### 场景: 情感分析服务

```python
# sentiment_app.py
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 加载情感分析模型 (Hugging Face)
sentiment_pipeline = pipeline('sentiment-analysis')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """
    分析文本情感
    
    请求格式: JSON {"text": "I love this product!"}
    响应格式: JSON
    """
    
    # 获取请求数据
    data = request.get_json()
    
    if not data or 'text' not in data:
        return jsonify({'error': '请提供 text 字段'}), 400
    
    text = data['text']
    
    if len(text) > 1000:
        return jsonify({'error': '文本长度不能超过 1000 字符'}), 400
    
    try:
        # 情感分析
        result = sentiment_pipeline(text)[0]
        
        response = {
            'success': True,
            'text': text,
            'sentiment': result['label'],  # POSITIVE 或 NEGATIVE
            'confidence': round(result['score'], 4)
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**测试:**
```python
import requests

response = requests.post(
    'http://localhost:5000/analyze',
    json={'text': 'I love this product! It is amazing!'}
)

print(response.json())
# {
#   "success": true,
#   "text": "I love this product! It is amazing!",
#   "sentiment": "POSITIVE",
#   "confidence": 0.9998
# }
```

---

## 五、批量预测 API

### 场景: 一次处理多个样本

```python
@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    批量预测
    
    请求格式: JSON {"texts": ["text1", "text2", ...]}
    响应格式: JSON
    """
    
    data = request.get_json()
    
    if not data or 'texts' not in data:
        return jsonify({'error': '请提供 texts 字段'}), 400
    
    texts = data['texts']
    
    if not isinstance(texts, list):
        return jsonify({'error': 'texts 必须是列表'}), 400
    
    if len(texts) > 100:
        return jsonify({'error': '一次最多处理 100 条'}), 400
    
    try:
        # 批量处理
        results = []
        for text in texts:
            result = sentiment_pipeline(text)[0]
            results.append({
                'text': text,
                'sentiment': result['label'],
                'confidence': round(result['score'], 4)
            })
        
        response = {
            'success': True,
            'count': len(results),
            'results': results
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
```

---

## 六、错误处理和验证

### 完善的错误处理

```python
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 全局错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '接口不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f'服务器错误: {error}')
    return jsonify({'error': '服务器内部错误'}), 500


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': '请求方法不允许'}), 405


# 请求前验证
@app.before_request
def validate_request():
    """验证所有请求"""
    if request.content_type and 'application/json' in request.content_type:
        if request.content_length and request.content_length > 1024 * 1024:  # 1MB
            return jsonify({'error': '请求体太大'}), 413


# 请求后记录日志
@app.after_request
def log_request(response):
    logger.info(f'{request.method} {request.path} - {response.status_code}')
    return response
```

---

## 七、API 文档

### 方法1: 手动编写文档

```python
@app.route('/docs', methods=['GET'])
def api_docs():
    """API 文档"""
    
    docs = """
    # 情感分析 API
    
    ## 接口列表
    
    ### 1. 健康检查
    - **URL**: /health
    - **方法**: GET
    - **响应**: {"status": "healthy"}
    
    ### 2. 情感分析
    - **URL**: /analyze
    - **方法**: POST
    - **请求体**: {"text": "要分析的文本"}
    - **响应**: 
      ```json
      {
        "success": true,
        "text": "原文本",
        "sentiment": "POSITIVE",
        "confidence": 0.9998
      }
      ```
    
    ### 3. 批量分析
    - **URL**: /batch_predict
    - **方法**: POST
    - **请求体**: {"texts": ["文本1", "文本2"]}
    - **响应**: 
      ```json
      {
        "success": true,
        "count": 2,
        "results": [...]
      }
      ```
    
    ## 错误码
    - 400: 请求参数错误
    - 404: 接口不存在
    - 500: 服务器错误
    """
    
    return docs, 200, {'Content-Type': 'text/plain'}
```

### 方法2: 使用 Swagger (推荐)

安装 Flask-RESTX:
```bash
pip install flask-restx
```

```python
from flask import Flask
from flask_restx import Api, Resource, fields

app = Flask(__name__)
api = Api(app, version='1.0', title='情感分析 API', description='简单的情感分析服务')

# 定义数据模型
ns = api.namespace('analyze', description='情感分析操作')

sentiment_model = api.model('SentimentRequest', {
    'text': fields.String(required=True, description='要分析的文本')
})

result_model = api.model('SentimentResult', {
    'success': fields.Boolean,
    'text': fields.String,
    'sentiment': fields.String,
    'confidence': fields.Float
})

@ns.route('/')
class SentimentAnalysis(Resource):
    @ns.doc('分析文本情感')
    @ns.expect(sentiment_model)
    @ns.marshal_with(result_model)
    def post(self):
        """分析文本情感"""
        data = api.payload
        text = data['text']
        
        result = sentiment_pipeline(text)[0]
        
        return {
            'success': True,
            'text': text,
            'sentiment': result['label'],
            'confidence': round(result['score'], 4)
        }


if __name__ == '__main__':
    app.run(debug=True)
```

**访问:** `http://localhost:5000` 会自动显示 Swagger UI,可以在线测试 API!

---

## 八、性能优化

### 技巧1: 模型预加载

```python
# ❌ 不好: 每次请求都加载模型
@app.route('/predict', methods=['POST'])
def predict():
    model = load_model('model.pth')  # 很慢!
    # ...

# ✅ 好: 启动时加载一次
model = load_model('model.pth')  # 全局变量

@app.route('/predict', methods=['POST'])
def predict():
    # 直接使用已加载的模型
    # ...
```

### 技巧2: 使用 Gunicorn (生产环境)

Flask 自带的服务器不适合生产环境,要用 Gunicorn:

```bash
pip install gunicorn
```

**运行:**
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

- `-w 4`: 4 个工作进程
- `-b 0.0.0.0:5000`: 绑定地址和端口
- `app:app`: 应用模块和应用对象

### 技巧3: 缓存结果

```python
from flask_caching import Cache

app.config['CACHE_TYPE'] = 'simple'
cache = Cache(app)

@app.route('/predict', methods=['POST'])
@cache.cached(timeout=300, query_string=True)  # 缓存 5 分钟
def predict():
    # 相同的输入会直接返回缓存结果
    # ...
```

---

## 九、完整项目结构

```
flask_api_project/
├── app.py              # Flask 主应用
├── model.py            # 模型定义和加载
├── requirements.txt    # 依赖列表
├── model.pth           # 训练好的模型
├── tests/              # 测试文件
│   └── test_api.py
└── README.md           # 项目说明
```

**requirements.txt:**
```
flask==2.3.0
torch==2.0.0
Pillow==9.5.0
torchvision==0.15.0
gunicorn==20.1.0
```

---

## 十、本章小结

### 核心要点

✅ **Flask 基础:**
- 创建应用: `app = Flask(__name__)`
- 定义路由: `@app.route('/path')`
- 运行服务器: `app.run()`

✅ **API 设计:**
- GET: 获取数据
- POST: 提交数据/预测
- 返回 JSON 格式
- 适当的错误码

✅ **最佳实践:**
- 模型预加载 (不要每次请求都加载)
- 完善的错误处理
- 输入验证
- 日志记录
- 使用 Gunicorn 部署

✅ **工具推荐:**
- Postman: 测试 API
- Swagger: 自动生成文档
- Gunicorn: 生产服务器

---

## 🎯 下一步

Flask 很简单,但性能有限。接下来学习 **FastAPI**,它更快、更现代:

- [Q4](./Day27-Q4%20-%20用%20FastAPI%20构建高性能%20API.md): FastAPI 详解
- [Q5](./Day27-Q5%20-%20Docker%20容器化部署.md): Docker 容器化
- [Q6](./Day27-Q6%20-%20云平台部署实战.md): 部署到云端

**继续前进!** 🚀
