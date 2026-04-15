# Day20-Q4 - 预训练模型和应用详解 + Day20-Q5 - 语音识别实战

> **难度等级：** ⭐⭐⭐⭐ | **预计用时：** 80-90 分钟

## 🎯 核心内容

### Q4: 预训练模型和应用

**主流预训练模型：**

1. **Whisper (OpenAI)**
   - 多语言支持（99种）
   - 68万小时训练数据
   - 多种尺寸（tiny到large）
   - 开箱即用

2. **Wav2Vec 2.0 (Facebook)**
   - 自监督预训练
   - 少样本微调效果好
   - 适合低资源场景

3. **HuBERT (Facebook)**
   - 隐藏单元聚类
   - 更好的表征学习
   - 性能优于 Wav2Vec

4. **Conformer (Google)**
   - CNN + Transformer
   - SOTA 性能
   - 生产级应用

**应用场景：**

- 语音转文字（转录）
- 实时字幕生成
- 会议记录
- 客服质检
- 语音搜索
- 无障碍辅助

---

### Q5: 语音识别实战

**完整项目示例：**

```python
"""
实时语音转录系统

功能：
→ 麦克风录音
→ 实时转录
→ 显示文字
→ 保存记录

技术栈：
→ Whisper
→ PyAudio
→ Tkinter/Streamlit
"""

import whisper
import pyaudio
import numpy as np
from datetime import datetime

class RealTimeTranscriber:
    """实时转录器"""
    
    def __init__(self, model_size="base"):
        # 加载模型
        self.model = whisper.load_model(model_size)
        
        # 音频配置
        self.sample_rate = 16000
        self.chunk_size = 4096
        
        print(f"✓ 模型加载完成: {model_size}")
    
    def record_audio(self, duration=5):
        """录制音频"""
        
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        print("开始录音...")
        frames = []
        
        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            frames.append(data)
        
        print("录音完成")
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # 转换为 numpy
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        return audio_data
    
    def transcribe(self, audio_data):
        """转录音频"""
        
        result = self.model.transcribe(audio_data)
        
        return result['text']
    
    def run(self):
        """运行转录系统"""
        
        print("=" * 50)
        print("🎤 实时语音转录系统")
        print("=" * 50)
        print("\n按 Ctrl+C 退出\n")
        
        try:
            while True:
                # 录音
                audio = self.record_audio(duration=5)
                
                # 转录
                text = self.transcribe(audio)
                
                # 显示
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {text}\n")
                
                # 保存
                with open('transcript.txt', 'a', encoding='utf-8') as f:
                    f.write(f"[{timestamp}] {text}\n")
        
        except KeyboardInterrupt:
            print("\n\n✓ 转录已保存到 transcript.txt")


# 运行
if __name__ == "__main__":
    transcriber = RealTimeTranscriber(model_size="base")
    transcriber.run()
```

**部署到 Web：**

```python
"""
使用 Streamlit 部署 Web 应用

安装：
pip install streamlit whisper

运行：
streamlit run app.py
"""

import streamlit as st
import whisper
import tempfile

st.title("🎤 语音转录应用")

# 上传音频
uploaded_file = st.file_uploader("上传音频文件", type=['wav', 'mp3', 'm4a'])

if uploaded_file:
    # 保存临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
        f.write(uploaded_file.read())
        temp_path = f.name
    
    # 选择模型
    model_size = st.selectbox(
        "选择模型",
        ["tiny", "base", "small", "medium", "large"]
    )
    
    if st.button("开始转录"):
        with st.spinner("转录中..."):
            # 加载模型
            model = whisper.load_model(model_size)
            
            # 转录
            result = model.transcribe(temp_path)
            
            # 显示结果
            st.success("转录完成！")
            st.text_area("转录结果", result['text'], height=300)
            
            # 下载
            st.download_button(
                "下载文本",
                result['text'],
                file_name="transcript.txt"
            )
```

**性能优化技巧：**

```python
"""
优化策略

1. 模型量化
   → FP16 推理
   → INT8 量化
   → 速度提升 2-3x

2. 批处理
   → 多个音频一起处理
   → 提高吞吐量

3. 缓存
   → 常用短语缓存
   → 减少重复计算

4. 流式处理
   → 分块转录
   → 降低延迟

5. GPU 加速
   → CUDA 支持
   → 大幅提速
"""

# GPU 加速示例
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)

print(f"✓ 使用设备: {device}")
print(f"  → GPU: 快 5-10x")
print(f"  → CPU: 慢但通用")
```

**评估指标：**

```python
"""
评估 ASR 性能

指标：
1. WER (Word Error Rate)
   → 词错误率
   → 越低越好
   
2. CER (Character Error Rate)
   → 字符错误率
   → 中文常用
   
3. RTF (Real Time Factor)
   → 实时因子
   → < 1 表示实时

计算：
WER = (S + D + I) / N
S: 替换数
D: 删除数
I: 插入数
N: 总词数
"""

def calculate_wer(reference, hypothesis):
    """计算 WER"""
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # 编辑距离
    from Levenshtein import distance
    edits = distance(ref_words, hyp_words)
    
    wer = edits / len(ref_words)
    
    return wer


print("=" * 50)
print("🎯 评估指标")
print("=" * 50)

print("""
常用指标:

1. WER (词错误率)
   → 英文标准
   → < 5% 优秀
   → < 10% 良好

2. CER (字符错误率)
   → 中文标准
   → < 3% 优秀
   → < 8% 良好

3. RTF (实时因子)
   → < 1: 实时
   → < 0.5: 超实时

基准数据集:
→ LibriSpeech (英文)
→ AISHELL (中文)
→ Common Voice (多语言)
""")
```

---

## 💡 实际应用案例

### 案例 1：在线教育平台

```python
"""
应用场景：
→ 课程视频自动字幕
→ 实时翻译
→  searchable 内容

价值：
→ 提高可访问性
→ 改善学习体验
→ SEO 优化
"""

print("在线教育应用:")
print("  ✓ 自动字幕")
print("  ✓ 多语言翻译")
print("  ✓ 内容搜索")
print("  ✓ 学习笔记")
```

### 案例 2：医疗病历系统

```python
"""
应用场景：
→ 医生口述病历
→ 自动转录
→ 结构化存储

要求：
→ 高准确率
→ 医学术语
→ 隐私保护
"""

print("医疗病历系统:")
print("  ✓ 口述录入")
print("  ✓ 术语识别")
print("  ✓ 隐私加密")
print("  ✓ 合规存储")
```

---

## ❌ 常见错误

### 错误 1：忽视口音和方言

**问题：**
- 模型在标准普通话上好
- 口音/方言效果差

**解决：**
```python
# 使用多语言模型
model = whisper.load_model("large")
# 或微调定制模型
fine_tune_on_dialect_data()
```

---

### 错误 2：忽略背景噪声

**问题：**
- 嘈杂环境识别率低

**解决：**
```python
# 降噪预处理
denoised = remove_noise(audio)
# 或使用鲁棒模型
model = whisper.load_model("large")  # 更鲁棒
```

---

## 🔍 总结

```python
print("=" * 50)
print("🎯 Day20 语音识别总结")
print("=" * 50)

print("""
核心知识点：

1. 信号处理:
   ✓ 采样、频谱
   ✓ 梅尔频谱
   ✓ MFCC 特征

2. 技术演进:
   ✓ HMM-GMM → CTC
   ✓ Attention → Transformer
   ✓ Conformer SOTA

3. 端到端模型:
   ✓ DeepSpeech
   ✓ Wav2Vec 2.0
   ✓ Whisper
   ✓ Conformer

4. 实战应用:
   ✓ 实时转录
   ✓ Web 部署
   ✓ 性能优化
   ✓ 评估指标

5. 最佳实践:
   ✓ 选择合适模型
   ✓ 数据预处理
   ✓ 降噪增强
   ✓ 监控优化

记住：
→ 语音识别已成熟
→ 选择合适的工具
→ 注重用户体验
→ 持续改进优化
""")

print("\n🎊 恭喜！Day20 语音识别基础全部完成！")
print("接下来准备 Day21 Week3 综合项目！")
```

---

## 📊 关键要点

| 主题 | 核心内容 | 重要性 |
|------|---------|--------|
| **信号处理** | 采样、MFCC | ⭐⭐⭐⭐⭐ |
| **技术演进** | CTC、Attention | ⭐⭐⭐⭐⭐ |
| **端到端模型** | Whisper、Conformer | ⭐⭐⭐⭐⭐ |
| **实战应用** | 转录、部署 | ⭐⭐⭐⭐⭐ |

**金句总结：**
> 语音识别全掌握，信号模型加实战；  
> Whisper 易用 Conformer 精，应用落地创 value！

---

## 💪 练习建议

### 基础练习
□ 理解核心概念
□ 使用预训练模型
□ 简单转录测试

### 进阶练习
□ 开发转录应用
□ 优化性能
□ 部署到 Web

### 高阶练习
□ 微调定制模型
□ 处理特殊场景
□ 生产级部署

---

## 🎯 自我评估

**完成度检查：**
- [ ] 我理解语音处理
- [ ] 我知道技术演进
- [ ] 我会使用模型
- [ ] 我能开发应用
- [ ] 我能部署优化

**掌握程度：** ⭐⭐⭐⭐⭐（涂色自评）

---

**📅 学习时间：** ______分钟  
**💬 输出次数：** ______次  
**✨ 最大收获：** _________________________________

---

> **记住：** 学以致用最重要！  
> **动手实践，创造价值！** 💪

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
