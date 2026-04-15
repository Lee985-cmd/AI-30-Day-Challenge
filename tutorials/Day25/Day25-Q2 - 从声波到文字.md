# Day25-Q2 - 从声波到文字

## 🎵 完整的语音识别流程

### 整体架构

```
原始音频
  ↓
[1] 预处理
  ↓
[2] 特征提取
  ↓
[3] 声学模型
  ↓
[4] 解码器
  ↓
[5] 语言模型
  ↓
识别文本
```

让我们逐步详解每个环节！

## 🔧 第一步：预处理

### 1. 读取音频文件

```python
import librosa
import numpy as np

# 加载音频文件
audio_path = "speech.wav"
signal, sample_rate = librosa.load(audio_path, sr=16000)

print(f"采样率: {sample_rate} Hz")
print(f"时长: {len(signal) / sample_rate:.2f} 秒")
print(f"采样点数: {len(signal)}")
print(f"数据类型: {signal.dtype}")
print(f"振幅范围: [{signal.min():.3f}, {signal.max():.3f}]")
```

**常见音频格式:**
```
WAV: 无损，未压缩
MP3: 有损压缩，文件小
FLAC: 无损压缩
AAC: 高效压缩

语音识别推荐:
- 格式: WAV 或 FLAC
- 采样率: 16kHz (标准)
- 位深度: 16bit
- 声道: 单声道 (Mono)
```

### 2. 预加重 (Pre-emphasis)

**目的：** 增强高频信号

```python
def pre_emphasis(signal, coefficient=0.97):
    """
    预加重滤波
    
    原理:
    y[t] = x[t] - α * x[t-1]
    
    作用:
    - 提升高频部分
    - 平衡频谱
    - 提高信噪比
    """
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])

# 应用预加重
emphasized_signal = pre_emphasis(signal)

print(f"原始信号能量: {np.sum(signal**2):.2f}")
print(f"预加重后能量: {np.sum(emphasized_signal**2):.2f}")
```

**为什么需要预加重？**
```
人声特点:
- 低频能量强 (基频)
- 高频能量弱 (谐波)

问题:
- 高频信息容易被忽略
- 影响音素识别

解决:
✓ 预加重提升高频
✓ 平衡频谱分布
✓ 改善识别效果
```

### 3. 分帧 (Framing)

**目的：** 将连续信号切成小段

```python
def frame_signal(signal, frame_length=0.025, frame_stride=0.01, sample_rate=16000):
    """
    分帧处理
    
    参数:
    - frame_length: 帧长 25ms
    - frame_stride: 帧移 10ms
    - sample_rate: 采样率 16kHz
    
    返回:
    - frames: 分帧后的信号
    """
    frame_len = int(frame_length * sample_rate)  # 400 个点
    frame_step = int(frame_stride * sample_rate)  # 160 个点
    signal_len = len(signal)
    
    # 计算帧数
    num_frames = int(np.ceil((signal_len - frame_len) / frame_step)) + 1
    
    # 补零
    pad_len = num_frames * frame_step + frame_len - signal_len
    if pad_len > 0:
        signal = np.append(signal, np.zeros(pad_len))
    
    # 提取帧
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    
    frames = signal[indices.astype(np.int32)]
    
    return frames

# 分帧
frames = frame_signal(signal)
print(f"总帧数: {frames.shape[0]}")
print(f"每帧长度: {frames.shape[1]} 个点")
print(f"第一帧形状: {frames[0].shape}")
```

**为什么要分帧？**
```
原因:
1. 短时平稳性
   - 语音在短时间 (20-40ms) 内近似平稳
   - 可以假设统计特性不变

2. 便于处理
   - 逐帧提取特征
   - 降低计算复杂度

3. 重叠设计
   - 帧长 25ms，帧移 10ms
   - 重叠 15ms (60%)
   - 避免边界信息丢失
```

### 4. 加窗 (Windowing)

**目的：** 减少频谱泄漏

```python
def apply_window(frames, window_type='hamming'):
    """
    加窗处理
    
    常用窗口:
    - Hamming: 最常用
    - Hanning: 类似 Hamming
    - Blackman: 更好的旁瓣抑制
    """
    if window_type == 'hamming':
        window = np.hamming(frames.shape[1])
    elif window_type == 'hanning':
        window = np.hanning(frames.shape[1])
    else:
        window = np.blackman(frames.shape[1])
    
    # 逐帧加窗
    windowed_frames = frames * window
    
    return windowed_frames

# 加窗
windowed_frames = apply_window(frames)

print(f"窗口函数形状: {window.shape}")
print(f"加窗前能量: {np.sum(frames[0]**2):.2f}")
print(f"加窗后能量: {np.sum(windowed_frames[0]**2):.2f}")
```

**为什么要加窗？**
```
问题:
- 截断信号会产生突变
- 导致频谱泄漏
- 影响频率分析

解决:
✓ 使用平滑窗口
✓ 两端渐变为 0
✓ 减少边界效应

Hamming 窗口公式:
w[n] = 0.54 - 0.46 * cos(2πn / (N-1))

特点:
- 主瓣宽度适中
- 旁瓣衰减好
- 计算简单
```

## 📊 第二步：特征提取

### 1. FFT (快速傅里叶变换)

**目的：** 时域 → 频域

```python
def compute_fft(frames, n_fft=512):
    """
    计算 FFT
    
    参数:
    - frames: 分帧加窗后的信号
    - n_fft: FFT 点数
    
    返回:
    - magnitude_spectrum: 幅度谱
    """
    # 计算 FFT
    fft_result = np.fft.rfft(frames, n=n_fft)
    
    # 计算幅度谱
    magnitude_spectrum = np.abs(fft_result)
    
    return magnitude_spectrum

# 计算 FFT
magnitude_spectra = compute_fft(windowed_frames)

print(f"FFT 结果形状: {magnitude_spectra.shape}")
print(f"频率分辨率: {16000 / 512:.2f} Hz/bin")
```

**FFT 的作用:**
```
时域信号:
- 横轴: 时间
- 纵轴: 振幅
- 难以看出频率成分

频域信号:
- 横轴: 频率
- 纵轴: 能量
- 清晰显示频率分布

示例:
元音 "a" 的频谱:
- F1 (第一共振峰): ~700Hz
- F2 (第二共振峰): ~1200Hz
- F3 (第三共振峰): ~2500Hz

这些共振峰是区分不同元音的关键!
```

### 2. 梅尔滤波器组 (Mel Filterbank)

**目的：** 模拟人耳听觉

```python
def mel_filterbank(n_filters=40, n_fft=512, sample_rate=16000):
    """
    创建梅尔滤波器组
    
    梅尔刻度:
    - 模拟人耳非线性感知
    - 低频分辨率高
    - 高频分辨率低
    
    公式:
    mel = 2595 * log10(1 + Hz/700)
    Hz = 700 * (10^(mel/2595) - 1)
    """
    
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)
    
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    # 梅尔刻度范围
    low_freq_mel = hz_to_mel(0)
    high_freq_mel = hz_to_mel(sample_rate // 2)
    
    # 均匀划分梅尔刻度
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    
    # 转换回 Hz
    hz_points = mel_to_hz(mel_points)
    
    # 转换为 FFT bin
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    # 创建滤波器
    fbank = np.zeros((n_filters, n_fft // 2 + 1))
    
    for m in range(1, n_filters + 1):
        f_left = bins[m - 1]
        f_center = bins[m]
        f_right = bins[m + 1]
        
        # 上升沿
        for k in range(f_left, f_center):
            fbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        
        # 下降沿
        for k in range(f_center, f_right):
            fbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    
    return fbank

# 创建滤波器组
fbank = mel_filterbank()

print(f"滤波器组形状: {fbank.shape}")
print(f"滤波器数量: {fbank.shape[0]}")
print(f"频率 bin 数量: {fbank.shape[1]}")
```

**为什么用梅尔刻度？**
```
人耳特性:
- 对低频敏感 (分辨力强)
- 对高频不敏感 (分辨力弱)

线性刻度:
0-1000Hz: 1000Hz 范围
1000-2000Hz: 1000Hz 范围
→ 相同的分辨率

梅尔刻度:
0-1000Hz: 更多滤波器
1000-2000Hz: 较少滤波器
→ 符合人耳感知

效果:
✓ 更符合听觉特性
✓ 提高识别准确率
✓ 减少冗余信息
```

### 3. 计算 Fbank 特征

```python
def compute_fbank(magnitude_spectra, fbank):
    """
    计算滤波器组能量特征
    
    步骤:
    1. 应用滤波器组
    2. 取对数
    3. 归一化 (可选)
    """
    # 应用滤波器组
    fbank_features = np.dot(magnitude_spectra, fbank.T)
    
    # 避免 log(0)
    fbank_features = np.where(fbank_features == 0, np.finfo(float).eps, fbank_features)
    
    # 取对数
    log_fbank = np.log(fbank_features)
    
    return log_fbank

# 计算 Fbank
log_fbank = compute_fbank(magnitude_spectra, fbank)

print(f"Fbank 特征形状: {log_fbank.shape}")
print(f"帧数: {log_fbank.shape[0]}")
print(f"特征维度: {log_fbank.shape[1]}")
```

### 4. MFCC (梅尔频率倒谱系数)

**目的：** 去相关，压缩特征

```python
def compute_mfcc(log_fbank, num_ceps=13):
    """
    计算 MFCC
    
    步骤:
    1. DCT (离散余弦变换)
    2. 取前 13 个系数
    3. 可选: 添加 delta 和 delta-delta
    """
    from scipy.fftpack import dct
    
    # DCT 变换
    mfcc = dct(log_fbank, type=2, axis=1, norm='ortho')
    
    # 取前 num_ceps 个系数
    mfcc = mfcc[:, :num_ceps]
    
    return mfcc

# 计算 MFCC
mfcc_features = compute_mfcc(log_fbank)

print(f"MFCC 特征形状: {mfcc_features.shape}")
print(f"特征维度: {mfcc_features.shape[1]}")

# 可视化 MFCC
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.imshow(mfcc_features.T, aspect='auto', origin='lower')
plt.title('MFCC 特征图')
plt.xlabel('时间帧')
plt.ylabel('MFCC 系数')
plt.colorbar()
plt.tight_layout()
plt.show()
```

**MFCC vs Fbank:**
```
Fbank:
- 40 维特征
- 相关性较强
- 深度学习常用

MFCC:
- 13 维特征
- 去相关 (DCT)
- 传统方法常用
- 压缩表示

选择:
✓ 深度学习: Fbank (保留更多信息)
✓ 传统 GMM-HMM: MFCC (去相关)
✓ 现代端到端: Fbank 或直接频谱
```

## 🎯 第三步：声学模型

### 传统方法：GMM-HMM

```python
"""
GMM-HMM 架构:

HMM (隐马尔可夫模型):
- 状态: 音素 (phoneme)
- 转移: 状态间跳转概率
- 观测: 声学特征

GMM (高斯混合模型):
- 建模每个状态的观测概率
- P(特征 | 状态)

工作流程:
1. 训练阶段
   - 对齐: 音频 ↔ 音素
   - 估计: GMM 参数
   - 优化: Baum-Welch 算法

2. 识别阶段
   - Viterbi 解码
   - 找最优状态序列
   - 输出音素序列

局限:
❌ 需要强制对齐
❌ 独立假设过强
❌ 表达能力有限
"""
```

### 深度学习方法：DNN-HMM

```python
"""
DNN-HMM 改进:

用 DNN 替代 GMM:
- 输入: 声学特征 (MFCC/Fbank)
- 输出: 音素后验概率 P(音素 | 特征)

优势:
✓ 更强的建模能力
✓ 自动学习特征
✓ 性能提升 20-30%

架构:
Input (40维 Fbank)
  ↓
Hidden Layer 1 (1024 neurons)
  ↓
Hidden Layer 2 (1024 neurons)
  ↓
Hidden Layer 3 (1024 neurons)
  ↓
Output (softmax, 音素数量)

训练:
- 交叉熵损失
- 反向传播
- SGD/Adam 优化
"""
```

### 现代方法：Transformer/Conformer

```python
"""
Transformer ASR:

架构:
Audio → CNN (降采样) → Transformer Encoder → Linear → CTC/Attention

优势:
✓ Self-Attention 捕捉长距离依赖
✓ 并行训练，速度快
✓ 端到端优化

Conformer (当前最佳):

结合 CNN + Transformer:
- CNN: 局部特征提取
- Transformer: 全局上下文

结构:
Conv Module → Attention Module → Feed Forward → Conv Module

性能:
- LibriSpeech test-clean: 2.3% WER
- 超越纯 Transformer
- 成为工业界标准
"""
```

## 🔍 第四步：解码器

### Beam Search 解码

```python
"""
Beam Search 原理:

目标: 找最优词序列
max P(W|A) = max P(A|W) * P(W)

算法:
1. 初始化: beam = [<start>]
2. 扩展: 对每个候选，生成 top-k 下一个词
3. 剪枝: 保留总分最高的 beam_size 个
4. 重复: 直到所有候选结束

示例 (beam_size=3):

步骤 1:
候选: ["我"]

步骤 2: 扩展
- "我想" (score=0.8)
- "我要" (score=0.7)
- "我是" (score=0.6)
- ... (其他更低)

保留 top-3: ["我想", "我要", "我是"]

步骤 3: 继续扩展
- "我想去" (0.8 * 0.9 = 0.72)
- "我想要" (0.8 * 0.6 = 0.48)
- "我要去" (0.7 * 0.85 = 0.595)
- ...

最终选择最高分的完整序列
"""
```

### CTC (Connectionist Temporal Classification)

```python
"""
CTC 解决的问题:

传统方法需要:
- 精确的时间对齐
- 知道每个音素的起止时间
- 标注成本高

CTC 的创新:
- 无需对齐
- 允许输入输出长度不同
- 引入 blank 符号

工作原理:

输入 (音频帧): [a, a, _, b, b, b, _, c]
输出 (去除重复和blank): [a, b, c]

路径:
"a_a_bb_c" → "abc"
"aa__bbc" → "abc"
"aaa_b_c" → "abc"

CTC 损失:
对所有能映射到目标的路径求和
L = -log(Σ P(path))

优势:
✓ 无需对齐
✓ 简化训练
✓ 端到端可行
"""
```

### Attention 机制

```python
"""
Attention-based ASR:

Listen, Attend and Spell (LAS):

Encoder:
- 提取音频特征
- 输出隐藏状态

Attention:
- 动态关注不同时间步
- 软对齐机制

Decoder:
- 自回归生成文本
- 类似语言模型

流程:
Audio → Encoder → Context Vector → Decoder → Text

优势:
✓ 自动学习对齐
✓ 端到端训练
✓ 性能优秀

局限:
❌ 必须完整输入才能开始输出
❌ 延迟较高
❌ 不适合流式识别

改进:
- Chunk-based Attention
- Monotonic Attention
- Transducer
"""
```

## 📝 第五步：语言模型

### N-gram 语言模型

```python
"""
N-gram 原理:

P(w1, w2, ..., wn) ≈ Π P(wi | wi-n+1, ..., wi-1)

Bigram (n=2):
P("去银行") = P("去"|"我") * P("银"|"去") * P("行"|"银")

Trigram (n=3):
P("去银行") = P("去"|"我想") * P("银"|"想去") * P("行"|"去银")

训练:
- 统计 n-gram 频率
- 平滑处理 (避免零概率)
- Kneser-Ney 平滑

应用:
- 纠正同音词
- 评估句子合理性
- 约束解码空间
"""
```

### Neural Language Model

```python
"""
神经语言模型:

RNN-LM:
Input → RNN/LSTM → Softmax → P(next word)

Transformer-LM:
Input → Transformer → Softmax → P(next word)

优势:
✓ 捕捉长距离依赖
✓ 更好的泛化能力
✓ 与声学模型联合训练

融合方式:

Shallow Fusion:
P_final = P_acoustic^α * P_LM^β

Deep Fusion:
- 联合训练
- 共享表示
- 端到端优化

效果:
- WER 降低 10-20%
- 显著提升流畅度
- 纠正语法错误
"""
```

## 🎓 完整流程总结

### 传统 Pipeline

```python
"""
完整流程代码框架:

class TraditionalASR:
    def __init__(self):
        self.acoustic_model = GMMHMM()
        self.language_model = NgramLM()
        self.decoder = WFSTDecoder()
    
    def recognize(self, audio):
        # 1. 预处理
        signal = load_audio(audio)
        emphasized = pre_emphasis(signal)
        frames = frame_signal(emphasized)
        windowed = apply_window(frames)
        
        # 2. 特征提取
        spectrum = compute_fft(windowed)
        fbank = apply_mel_filterbank(spectrum)
        mfcc = compute_mfcc(fbank)
        
        # 3. 声学评分
        acoustic_scores = self.acoustic_model.score(mfcc)
        
        # 4. 解码
        lattice = self.decoder.decode(acoustic_scores)
        
        # 5. 语言模型重打分
        best_path = self.language_model.rescore(lattice)
        
        return best_path.text
"""
```

### 端到端流程

```python
"""
端到端流程 (Whisper 风格):

class EndToEndASR:
    def __init__(self):
        self.model = WhisperModel()
        self.processor = WhisperProcessor()
    
    def recognize(self, audio):
        # 1. 音频预处理
        input_features = self.processor(audio)
        
        # 2. 编码
        encoder_output = self.model.encoder(input_features)
        
        # 3. 解码 (自回归)
        tokens = []
        for _ in range(max_length):
            decoder_output = self.model.decoder(tokens, encoder_output)
            next_token = argmax(decoder_output)
            tokens.append(next_token)
            if next_token == EOS:
                break
        
        # 4. 转换为文本
        text = self.processor.decode(tokens)
        
        return text

优势:
✓ 简洁优雅
✓ 联合优化
✓ 性能卓越
✓ 易于部署
"""
```

## 💡 关键要点

### 特征工程的重要性

```
好的特征 = 成功的一半

传统方法:
- 精心设计 MFCC
- 手工调整参数
- 领域知识重要

深度学习方法:
- 自动学习特征
- 端到端优化
- 但仍需合理预处理
```

### 模型选择的演进

```
2000s: GMM-HMM
2010s: DNN-HMM
2015s: LSTM/Attention
2020s: Transformer/Conformer
2022+: Whisper/Wav2Vec

趋势:
- 越来越简单
- 性能越来越好
- 数据越来越重要
```

### 实际系统的考虑

```
生产环境要求:

1. 实时性
   - RTF < 0.5
   - 低延迟

2. 准确性
   - WER < 10%
   - 鲁棒性强

3. 资源效率
   - 内存占用
   - CPU/GPU 使用

4. 可扩展性
   - 多语言支持
   - 领域适配
```

## 🚀 下一步

现在我们理解了从声波到文字的完整流程，接下来让我们深入了解端到端语音识别和 Whisper 模型。

---

**下一步：** [Day25-Q3 - 端到端语音识别](./Day25-Q3%20-%20端到端语音识别.md)
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
