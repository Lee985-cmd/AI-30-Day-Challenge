# Day28-Q2 - 隐私保护和数据安全

## 🔒 你的数据安全吗?

### 问题背景

你开发了一个健康预测 App,用户需要上传:
- 病历记录
- 基因数据
- 生活习惯
- 位置信息

**问题来了:**
- ❓ 这些数据存在哪里?
- ❓ 谁可以访问?
- ❓ 会不会被黑客窃取?
- ❓ 会不会被卖给广告商?
- ❓ 用户能删除自己的数据吗?

这就是 **隐私保护 (Privacy)** 和 **数据安全 (Data Security)** 问题。

---

## 一、为什么隐私很重要?

### 现实案例

#### 案例1: Facebook-Cambridge Analytica (2018)

**发生了什么:**
- Cambridge Analytica 获取了 8700 万 Facebook 用户数据
- 用于政治广告定向投放
- 影响了多国选举

**教训:**
- 个人数据可能被滥用
- 影响民主进程
- 需要更严格的数据保护

#### 案例2: 医疗数据泄露

**统计数据:**
- 2021年,美国医疗行业发生 700+ 起数据泄露
- 影响 4500 万人
- 平均每次泄露成本: $930 万

**后果:**
- 患者隐私暴露
- 身份盗窃风险
- 歧视性保险定价

#### 案例3: 人脸识别滥用

**例子:**
- Clearview AI: 从社交媒体抓取 30 亿张照片
- 建立面部识别数据库
- 卖给执法机构和公司
- 未经任何人同意

**问题:**
- 侵犯隐私权
- 大规模监控
- 缺乏监管

---

## 二、隐私威胁类型

### 威胁1: 数据泄露

**定义:** 未经授权访问敏感数据

**常见原因:**
- 弱密码
- SQL 注入攻击
- 内部人员泄密
- 钓鱼攻击

**防护:**
```python
# ✅ 密码哈希存储 (不要明文!)
import bcrypt

password = "user_password"
hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

# 验证时
bcrypt.checkpw(password.encode(), hashed)
```

### 威胁2: 模型逆向攻击

**定义:** 从模型输出推断训练数据

**例子:**
```python
# 攻击者查询模型
query = "What is John Smith's medical condition?"
model_output = model.predict(query)

# 通过多次查询,可能推断出 John 的隐私信息
```

**防护:**
- 差分隐私 (Differential Privacy)
- 限制查询频率
- 添加噪声

### 威胁3: 成员推断攻击

**定义:** 判断某条数据是否在训练集中

**风险:**
- 如果知道某人数据在"癌症患者"训练集中
- 就泄露了他的健康状况

**防护:**
- 正则化
- Dropout
- 差分隐私

### 威胁4: 数据重识别

**定义:** 匿名化数据重新关联到个人

**著名案例:**
- Netflix Prize 数据集"匿名化"
- 研究者结合 IMDb 数据
- 成功识别出具体用户

**教训:**
- 简单的匿名化不够
- 需要更强的隐私保护技术

---

## 三、隐私保护技术

### 技术1: 差分隐私 (Differential Privacy)

**核心思想:** 添加噪声,使得无法判断单个个体是否在数据集中

**大白话:**
```
原始数据: "张三得了癌症"
加噪声后: "大约 100±5 人得了癌症"

攻击者无法确定张三是否在数据集中
```

**实现:**
```python
import numpy as np

def add_laplace_noise(value, sensitivity, epsilon=1.0):
    """
    添加拉普拉斯噪声实现差分隐私
    
    sensitivity: 敏感度 (单个记录对结果的最大影响)
    epsilon: 隐私预算 (越小越隐私,但准确性越低)
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise

# 例子: 统计患病人数
true_count = 100
private_count = add_laplace_noise(true_count, sensitivity=1, epsilon=0.5)
print(f"真实值: {true_count}, 发布值: {private_count:.0f}")
```

**应用:**
- Apple: iOS 数据收集
- Google: Chrome 遥测
- US Census: 2020 人口普查

### 技术2: 联邦学习 (Federated Learning)

**核心思想:** 数据不出本地,只共享模型更新

**流程:**
```
1. 服务器发送全局模型到各设备
2. 每个设备用本地数据训练
3. 只上传模型梯度 (不是原始数据)
4. 服务器聚合梯度,更新全局模型
```

**优点:**
- 原始数据永不离开设备
- 保护用户隐私
- 符合 GDPR 等法规

**实现:**
```python
import tensorflow as tf
import tensorflow_federated as tff

# 定义模型
def create_keras_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

# 转换为 TFF 模型
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=train_data.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

# 联邦平均算法
federated_averaging = tff.learning.build_federated_averaging_process(model_fn)

# 训练
state = federated_averaging.initialize()
for round_num in range(100):
    state, metrics = federated_averaging.next(state, federated_train_data)
    print(f'Round {round_num}: {metrics}')
```

**应用:**
- Gboard 键盘预测
- 医疗影像分析
- 金融风控

### 技术3: 同态加密 (Homomorphic Encryption)

**核心思想:** 在加密数据上直接计算,结果解密后与明文计算相同

**比喻:**
```
传统方式:
数据 → 解密 → 计算 → 加密 → 结果

同态加密:
加密数据 → 计算 (密文) → 结果 (密文) → 解密 → 结果

全程数据都是加密的!
```

**库:**
```python
from tenseal import ts

# 创建上下文
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

# 加密
plain_vector = [1, 2, 3, 4]
encrypted_vector = ts.ckks_vector(context, plain_vector)

# 在加密数据上计算
result_encrypted = encrypted_vector + encrypted_vector
result = result_encrypted.decrypt()

print(f"原始: {plain_vector}")
print(f"加密后加法: {result}")  # [2, 4, 6, 8]
```

**缺点:**
- 计算慢 (比明文慢 100-1000 倍)
- 内存占用大

**应用:**
- 隐私保护机器学习
- 安全多方计算
- 私密信息查询

### 技术4: 数据脱敏

**方法:**

```python
import hashlib
import faker

fake = faker.Faker()

def anonymize_dataset(df):
    """数据脱敏"""
    
    df_anon = df.copy()
    
    # 1. 直接标识符: 删除或替换
    df_anon['name'] = [fake.name() for _ in range(len(df))]
    df_anon['email'] = [fake.email() for _ in range(len(df))]
    
    # 2. 准标识符: 泛化
    # 年龄: 25 → 20-30
    df_anon['age_range'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], 
                                   labels=['0-18', '19-30', '31-50', '51+'])
    df_anon.drop('age', axis=1, inplace=True)
    
    # 3. 敏感属性: 扰动
    df_anon['salary'] = df['salary'] * np.random.uniform(0.9, 1.1, len(df))
    
    # 4. 唯一标识符: 哈希
    df_anon['user_id'] = df['user_id'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
    )
    
    return df_anon
```

**K-匿名 (K-Anonymity):**

确保每条记录至少与其他 K-1 条记录不可区分

```python
from anonypy import Preserver

# 实现 k-匿名
preserver = Preserver(df, ['age_range', 'zipcode', 'gender'], ['disease'])
df_k_anonymous = preserver.preserve(k=5)  # 5-匿名
```

---

## 四、数据安全最佳实践

### 实践1: 加密存储

```python
from cryptography.fernet import Fernet

# 生成密钥 (只需一次)
key = Fernet.generate_key()
cipher = Fernet(key)

# 加密
data = b"sensitive information"
encrypted = cipher.encrypt(data)

# 解密
decrypted = cipher.decrypt(encrypted)

# ⚠️ 密钥要安全存储 (不要硬编码在代码中!)
# 使用环境变量或密钥管理服务
```

### 实践2: 访问控制

```python
from functools import wraps
from flask import request, jsonify
import jwt

def require_auth(role='user'):
    """身份验证装饰器"""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            token = request.headers.get('Authorization')
            
            if not token:
                return jsonify({'error': 'Token required'}), 401
            
            try:
                # 验证 token
                payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                
                # 检查角色
                if payload.get('role') != role and role != 'any':
                    return jsonify({'error': 'Insufficient permissions'}), 403
                
                # 添加到请求上下文
                request.user = payload
                
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401
            
            return f(*args, **kwargs)
        return decorated
    return decorator

# 使用
@app.route('/admin/data')
@require_auth(role='admin')
def get_sensitive_data():
    return jsonify({'data': 'sensitive'})
```

### 实践3: 审计日志

```python
import logging
from datetime import datetime

# 配置审计日志
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)

# 单独的文件处理器
file_handler = logging.FileHandler('audit.log')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
audit_logger.addHandler(file_handler)

def log_data_access(user_id, action, resource, success=True):
    """记录数据访问日志"""
    
    audit_logger.info({
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id,
        'action': action,  # read, write, delete
        'resource': resource,
        'success': success,
        'ip_address': request.remote_addr
    })

# 使用
@app.route('/api/users/<int:user_id>')
@require_auth()
def get_user(user_id):
    log_data_access(request.user['id'], 'read', f'user/{user_id}')
    # ...
```

### 实践4: 数据最小化

**原则:** 只收集和保留必要的数据

```python
# ❌ 不好: 收集过多数据
user_data = {
    'name': 'John',
    'email': 'john@example.com',
    'phone': '123-456-7890',
    'address': '123 Main St',
    'birthday': '1990-01-01',
    'ssn': '123-45-6789',  # 真的需要吗?
}

# ✅ 好: 只收集必要的
user_data = {
    'user_id': 'uuid-123',
    'email_hash': 'abc123...',  # 哈希而非明文
}
```

### 实践5: 定期删除

```python
from datetime import datetime, timedelta

def cleanup_old_data(retention_days=90):
    """删除超过保留期的数据"""
    
    cutoff_date = datetime.now() - timedelta(days=retention_days)
    
    # 删除旧记录
    deleted_count = db.users.delete_many({
        'last_active': {'$lt': cutoff_date},
        'account_type': 'inactive'
    }).deleted_count
    
    print(f"Deleted {deleted_count} old records")
    
    # 记录删除操作
    audit_logger.info(f"Cleanup: deleted {deleted_count} records before {cutoff_date}")
```

---

## 五、法律法规

### GDPR (欧盟通用数据保护条例)

**适用范围:** 处理欧盟公民数据的任何组织

**核心权利:**
1. **知情权**: 用户有权知道数据如何被使用
2. **访问权**: 用户可以索取自己的数据
3. **更正权**: 用户可以要求更正错误数据
4. **删除权** ("被遗忘权"): 用户可以要求删除数据
5. **可携带权**: 用户可以将数据转移到其他服务
6. **反对权**: 用户可以反对某些数据处理

**要求:**
- 默认隐私保护 (Privacy by Design)
- 数据泄露 72 小时内报告
- 任命数据保护官 (DPO)
- 违规罚款: 最高 €2000 万或全球营业额 4%

**合规示例:**
```python
@app.route('/api/user/data', methods=['GET'])
@require_auth()
def export_user_data():
    """GDPR: 数据导出"""
    
    user_id = request.user['id']
    
    # 收集用户所有数据
    user_data = {
        'profile': db.users.find_one({'id': user_id}),
        'activities': list(db.activities.find({'user_id': user_id})),
        'preferences': db.preferences.find_one({'user_id': user_id})
    }
    
    # 返回 JSON
    return jsonify(user_data)


@app.route('/api/user/delete', methods=['DELETE'])
@require_auth()
def delete_user_data():
    """GDPR: 删除用户数据"""
    
    user_id = request.user['id']
    
    # 删除所有相关数据
    db.users.delete_one({'id': user_id})
    db.activities.delete_many({'user_id': user_id})
    db.preferences.delete_one({'user_id': user_id})
    
    # 记录删除 (法律要求保留某些日志)
    audit_logger.info(f"User {user_id} requested data deletion")
    
    return jsonify({'status': 'deleted'})
```

### CCPA (加州消费者隐私法案)

**类似 GDPR,但针对加州居民**

**权利:**
- 知道收集了什么数据
- 知道数据是否被出售
- 选择不出售数据
- 删除个人数据

### 中国《个人信息保护法》

**2021年11月1日生效**

**要点:**
- 明确个人信息定义
- 要求用户同意
- 跨境数据传输限制
- 违规罚款: 最高 5000 万或营业额 5%

---

## 六、隐私设计原则

### 原则1: Privacy by Design

**7个基础原则:**
1. 主动而非被动
2. 默认隐私保护
3. 隐私嵌入设计
4. 全功能 (非零和)
5. 端到端安全
6. 可见性和透明度
7. 尊重用户隐私

**实践:**
```python
# ✅ 默认隐私保护
class UserSettings:
    def __init__(self):
        # 默认最严格的隐私设置
        self.data_sharing = False
        self.analytics_opt_in = False
        self.personalized_ads = False
        self.location_tracking = False
    
    def opt_in(self, feature):
        """用户主动选择加入"""
        setattr(self, feature, True)
        log_consent(feature)  # 记录同意
```

### 原则2: 数据生命周期管理

```
收集 → 存储 → 使用 → 共享 → 归档 → 删除
  ↓      ↓      ↓      ↓      ↓      ↓
最小化  加密   授权   合同   匿名   彻底
```

### 原则3: 透明度和控制

**给用户控制权:**
```python
@app.route('/privacy/settings', methods=['GET', 'POST'])
def privacy_settings():
    """隐私设置页面"""
    
    if request.method == 'POST':
        # 更新用户隐私偏好
        settings = request.json
        update_user_privacy(request.user['id'], settings)
        
        return jsonify({'status': 'updated'})
    
    else:
        # 显示当前设置
        settings = get_user_privacy(request.user['id'])
        return jsonify(settings)
```

**UI 示例:**
```
隐私设置
├─ 数据收集
│  ├─ [✓] 基本使用数据
│  ├─ [ ] 位置信息
│  └─ [ ] 个性化推荐
├─ 数据共享
│  ├─ [ ] 与合作伙伴共享
│  └─ [ ] 用于研究
└─ 数据保留
   └─ 自动删除: [30天 ▼]
```

---

## 七、本章小结

### 核心要点

✅ **隐私威胁:**
- 数据泄露
- 模型逆向攻击
- 成员推断攻击
- 数据重识别

✅ **保护技术:**
- 差分隐私 (添加噪声)
- 联邦学习 (数据不出本地)
- 同态加密 (密文计算)
- 数据脱敏 (K-匿名)

✅ **最佳实践:**
- 加密存储
- 访问控制
- 审计日志
- 数据最小化
- 定期删除

✅ **法律法规:**
- GDPR (欧盟)
- CCPA (加州)
- 个人信息保护法 (中国)

### 重要认知

⚠️ **隐私是基本权利:**
- 不是可有可无的功能
- 需要在设计阶段考虑
- 需要技术和法律双重保障

⚠️ **平衡创新和保护:**
- 过度保护阻碍创新
- 保护不足伤害用户
- 需要找到平衡点

---

## 🎯 下一步

理解了隐私保护,继续学习其他伦理问题:

- [Q3](./Day28-Q3%20-%20透明度和可解释性.md): AI 黑盒问题
- [Q4](./Day28-Q4%20-%20责任和安全.md): AI 出错了谁负责
- [Q5](./Day28-Q5%20-%20法律法规和监管.md): 各国 AI 法规详解
- [Q6](./Day28-Q6%20-%20AI 从业者的责任.md): 职业道德准则

**记住:** 保护用户隐私不仅是法律要求,更是道德责任! 🔒


---

## 📱 关于作者 & 获取更多资源

本教程由 **Lee（职场宝爸）** 创建，记录从零基础到独立完成 AI 项目的真实历程。

### 关注公众号，获取独家内容

**公众号名称：Lee 的成长日记**

微信搜索关注，获取：
- ✅ **AI 学习路线规划**：零基础如何系统学习 AI
- ✅ **项目实战源码**：完整可运行的项目代码
- ✅ **深度技术解析**：前沿技术原理 + 手写代码实现
- ✅ **职场成长心得**：一个宝爸的 AI 逆袭之路

**关注福利**：
- 回复「**路线**」→ 获取 30 天 AI 学习计划表
- 回复「**项目**」→ 获取 GitHub 项目源码合集
- 回复「**资料**」→ 获取零基础学习资源推荐

**扫码关注公众号**：

![公众号二维码](../../../images/logos/ewm.jpg)

### 其他平台

- 📂 **GitHub**：https://github.com/Lee985-cmd/AI-30Days-Challenge
- 📝 **CSDN 博客**：https://blog.csdn.net/m0_67081842
- 💬 **公众号**：微信搜索「Lee 的成长日记」

---

> 💡 **学习建议**
> 
> 如果本篇教程对你有帮助，欢迎：
> 1. **Star GitHub 项目**：https://github.com/Lee985-cmd/AI-30Days-Challenge
> 2. **关注公众号**获取更多独家内容
> 3. **留言交流**你的学习困惑
> 
> **一起学习，一起进步！** 🤝
