# 🎯 Day21: Week3 综合项目 - 完整的 AI 系统实战【真正零基础版】

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **三选一！做一个能写进简历的完整项目!**  
> **本教程：3 个项目方向 + 完整代码 + 部署指南**

---

## 📚 目录

1. [项目 A: 智能监控系统](#项目 a-智能监控系统)
2. [项目 B: 自动驾驶感知系统](#项目 b-自动驾驶感知系统)
3. [项目 C: 医学影像分析](#项目 c-医学影像分析)
4. [项目开发流程](#项目开发流程)
5. [答辩和展示](#答辩和展示)

---

## 🎯 项目 A: 智能监控系统

### 项目介绍

```python
"""
场景:
商场、公司、小区都需要监控
传统监控:
- 只能录像
- 出事了才回看
- 被动响应

智能监控:
✓ 实时识别人脸
✓ 统计人流量
✓ 发现异常行为
✓ 主动预警

这个项目能做进简历:
✓ 计算机视觉综合应用
✓ 有实际价值
✓ 效果直观
"""
```

### 功能需求

```python
"""
核心功能:

1. 人脸检测和识别
   - 检测画面中的人脸
   - 识别是不是熟人
   - 标记陌生人

2. 人数统计
   - 实时统计在场人数
   - 记录进出人流
   - 生成统计报表

3. 轨迹追踪
   - 跟踪每个人的移动路径
   - 分析活动区域
   - 发现异常行为

4. 告警系统
   - 发现黑名单人员报警
   - 区域入侵检测
   - 推送通知
"""
```

### 技术架构

```python
"""
技术栈:

【人脸检测】
- MTCNN: 高精度人脸检测
- RetinaFace: 更快的人脸检测

【人脸识别】
- FaceNet: 128 维特征向量
- ArcFace: 更准确的识别

【目标跟踪】
- DeepSORT: 多目标跟踪算法
- ByteTrack: 最新的跟踪方法

【系统集成】
- OpenCV: 视频处理
- Flask/FastAPI: Web 界面
- SQLite: 数据存储
"""
```

### 完整代码实现

```python
# ============================================================================
# 第一部分：导入库
# ============================================================================

import cv2
import numpy as np
import torch
from datetime import datetime
import sqlite3
import os

print("=" * 60)
print("智能监控系统 - 完整实现")
print("=" * 60)

# ============================================================================
# 第二部分：人脸检测模块
# ============================================================================

class FaceDetector:
    """人脸检测器"""
    
    def __init__(self):
        # 使用 OpenCV 的预训练模型
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ 人脸检测器加载成功")
    
    def detect(self, frame):
        """
        检测人脸
        
        参数:
        frame: 视频帧 (BGR 格式)
        
        返回:
        faces: 人脸位置列表 [(x,y,w,h), ...]
        """
        # 转灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测人脸
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # 图像缩放比例
            minNeighbors=5,       # 每个候选矩形至少保留几个邻居
            minSize=(30, 30)      # 最小人脸尺寸
        )
        
        return faces

# ============================================================================
# 第三部分：人脸识别模块 (简化版)
# ============================================================================

class FaceRecognizer:
    """人脸识别器"""
    
    def __init__(self):
        self.known_faces = {}  # 存储已知人脸 {name: embedding}
        print("✓ 人脸识别器初始化完成")
    
    def register_face(self, name, face_embedding):
        """注册人脸"""
        self.known_faces[name] = face_embedding
        print(f"✓ 已注册：{name}")
    
    def recognize(self, face_embedding, threshold=0.6):
        """
        识别人脸
        
        参数:
        face_embedding: 当前人脸特征
        threshold: 相似度阈值
        
        返回:
        name: 识别结果 (未知则返回"Unknown")
        confidence: 置信度
        """
        if not self.known_faces:
            return "Unknown", 0.0
        
        best_name = "Unknown"
        best_similarity = 0.0
        
        for name, known_emb in self.known_faces.items():
            # 计算余弦相似度
            similarity = np.dot(face_embedding, known_emb) / (
                np.linalg.norm(face_embedding) * np.linalg.norm(known_emb)
            )
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_name = name
        
        return best_name, best_similarity

# ============================================================================
# 第四部分：人数统计模块
# ============================================================================

class PeopleCounter:
    """人数计数器"""
    
    def __init__(self):
        self.count = 0
        self.enter_count = 0
        self.exit_count = 0
        print("✓ 人数计数器初始化完成")
    
    def update(self, faces):
        """更新人数统计"""
        self.count = len(faces)
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'current': self.count,
            'entered': self.enter_count,
            'exited': self.exit_count
        }

# ============================================================================
# 第五部分：数据库管理
# ============================================================================

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_name='surveillance.db'):
        self.db_name = db_name
        self.create_tables()
        print(f"✓ 数据库创建成功：{db_name}")
    
    def create_tables(self):
        """创建数据表"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        # 人员表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                is_blacklist BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 检测记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                camera_id TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons (id)
            )
        ''')
        
        # 告警记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_person(self, name, is_blacklist=False):
        """添加人员"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO persons (name, is_blacklist) VALUES (?, ?)',
                (name, is_blacklist)
            )
            conn.commit()
            person_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            # 已存在，查询 ID
            cursor.execute('SELECT id FROM persons WHERE name = ?', (name,))
            person_id = cursor.fetchone()[0]
        finally:
            conn.close()
        
        return person_id
    
    def log_detection(self, person_id, camera_id='cam1'):
        """记录检测"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO detections (person_id, camera_id) VALUES (?, ?)',
            (person_id, camera_id)
        )
        
        conn.commit()
        conn.close()
    
    def add_alert(self, alert_type, description):
        """添加告警"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT INTO alerts (alert_type, description) VALUES (?, ?)',
            (alert_type, description)
        )
        
        conn.commit()
        conn.close()
        print(f"⚠️ 告警：{alert_type} - {description}")

# ============================================================================
# 第六部分：主监控系统
# ============================================================================

class SmartSurveillanceSystem:
    """智能监控系统"""
    
    def __init__(self):
        # 初始化各模块
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.counter = PeopleCounter()
        self.db = DatabaseManager()
        
        # 注册一些已知人员
        self.register_known_faces()
        
        print("✓ 智能监控系统启动成功")
    
    def register_known_faces(self):
        """注册已知人脸 (示例)"""
        # 实际项目中这里应该加载真实的人脸特征
        # 为了演示，我们用随机向量模拟
        self.recognizer.register_face("张三", np.random.rand(128))
        self.recognizer.register_face("李四", np.random.rand(128))
        self.recognizer.register_face("王五", np.random.rand(128))
        
        # 添加黑名单
        self.db.add_person("坏人 A", is_blacklist=True)
        self.recognizer.register_face("坏人 A", np.random.rand(128))
    
    def process_frame(self, frame):
        """
        处理单帧图像
        
        参数:
        frame: 视频帧
        
        返回:
        processed_frame: 处理后的帧
        """
        # 1. 人脸检测
        faces = self.detector.detect(frame)
        
        # 2. 更新人数统计
        self.counter.update(faces)
        
        # 3. 绘制检测结果
        processed_frame = frame.copy()
        
        for i, (x, y, w, h) in enumerate(faces):
            # 画人脸框
            cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 这里应该做人脸识别
            # 为了演示，我们随机识别
            if i < 3:  # 前 3 个人脸假装认识
                names = ["张三", "李四", "王五"]
                name = names[i % len(names)]
                color = (0, 255, 0)  # 绿色=熟人
                
                # 检查是否黑名单
                if name == "坏人 A":
                    color = (0, 0, 255)  # 红色=黑名单
                    self.db.add_alert("BLACKLIST", f"发现黑名单人员：{name}")
                
                # 显示名字
                cv2.putText(processed_frame, name, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # 记录检测
                person_id = self.db.add_person(name)
                self.db.log_detection(person_id)
        
        # 显示人数
        stats = self.counter.get_stats()
        cv2.putText(processed_frame, f"Current: {stats['current']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return processed_frame
    
    def run(self, camera_id=0):
        """
        运行监控系统
        
        参数:
        camera_id: 摄像头 ID (0=默认摄像头)
        """
        print("\n正在启动监控...")
        print("按 'q' 键退出\n")
        
        # 打开摄像头
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ 无法打开摄像头!")
            return
        
        print("✓ 摄像头已打开")
        
        frame_count = 0
        
        while True:
            # 读取帧
            ret, frame = cap.read()
            
            if not ret:
                print("❌ 无法读取视频帧!")
                break
            
            # 处理帧
            processed_frame = self.process_frame(frame)
            
            # 显示结果
            cv2.imshow('Smart Surveillance System', processed_frame)
            
            # 每秒处理一次 (节省资源)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # 清理
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n✓ 监控结束，共处理 {frame_count} 帧")

# ============================================================================
# 第七部分：运行系统
# ============================================================================

if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║          智能监控系统 v1.0                            ║
║                                                       ║
║  功能:                                                ║
║  ✓ 实时人脸检测和识别                                ║
║  ✓ 人数统计                                          ║
║  ✓ 黑名单告警                                        ║
║  ✓ 数据记录和查询                                    ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝

使用说明:
1. 确保摄像头已连接
2. 运行程序会自动打开摄像头
3. 按 'q' 键退出
4. 数据保存在 surveillance.db

""")
    
    # 创建并运行系统
    system = SmartSurveillanceSystem()
    
    # 运行监控
    try:
        system.run(camera_id=0)  # 0=默认摄像头
    except KeyboardInterrupt:
        print("\n\n程序中断")
    except Exception as e:
        print(f"\n错误：{e}")
        print("\n提示:")
        print("  - 如果没有摄像头，可以用视频文件测试:")
        print("    cap = cv2.VideoCapture('test_video.mp4')")
        print("  - 确保安装了 opencv-python:")
        print("    pip install opencv-python")

# ============================================================================
# 第八部分：Web 界面 (可选扩展)
# ============================================================================

web_interface_code = """
# 用 FastAPI 创建 Web 管理界面

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io

app = FastAPI(title="智能监控系统")

@app.get("/")
async def root():
    return {"message": "智能监控系统 API"}

@app.get("/stats")
async def get_statistics():
    """获取统计数据"""
    return {
        "current_people": 0,
        "total_detected": 0,
        "alerts_count": 0
    }

@app.get("/alerts")
async def get_alerts():
    """获取告警列表"""
    return []

# 运行:
# uvicorn web_app:app --reload
"""

print("\n" + "=" * 60)
print("扩展功能")
print("=" * 60)
print("""
【可以添加的功能】

1. Web 管理界面
   - 实时查看监控
   - 查询历史记录
   - 管理黑白名单

2. 手机 APP
   - 接收告警推送
   - 远程查看监控
   - 语音对讲

3. 数据分析
   - 人流高峰时段
   - 热门区域分析
   - 行为模式识别

4. 云存储
   - 视频云端备份
   - 分布式存储
   - CDN 加速访问

5. 边缘计算
   - 本地预处理
   - 只上传关键帧
   - 降低带宽成本
""")

print("\n🎉 恭喜你完成了智能监控系统!")
print("\n下一步:")
print("  1. 用真实人脸数据训练识别模型")
print("  2. 添加更多功能 (轨迹追踪等)")
print("  3. 部署到实际环境")
print("  4. 写成博客分享经验")

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day20](../Day20/README.md)
- [→ Day22](../Day22/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*

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
