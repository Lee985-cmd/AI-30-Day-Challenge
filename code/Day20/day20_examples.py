"""
Day20 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day20_examples.py

注意: 某些代码可能需要安装额外的库
"""

# 导入必要的库
import sys
import os

# 尝试导入常用库
try:
    import numpy as np
except ImportError:
    print("提示: 需要安装 numpy: pip install numpy")
    np = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("提示: 需要安装 matplotlib: pip install matplotlib")
    plt = None

try:
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("提示: 需要安装 scikit-learn: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("提示: 需要安装 PyTorch: pip install torch torchvision")

print("=" * 60)
print("Day20 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

import torch
import whisper
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

print("=" * 60)
print("🎤 Whisper 语音识别实战")
print("=" * 60)

# ============================================================================
# 第 1 步：加载 Whisper 模型
# ============================================================================
print("\n【1. 加载 Whisper 模型】")

# 选择模型大小
# tiny: 39M, base: 74M, small: 244M, medium: 769M, large: 1550M
model_size = "base"

print(f"正在加载 {model_size} 模型...")
print("提示：第一次会自动下载，请耐心等待")

model = whisper.load_model(model_size)

print(f"✓ Whisper {model_size} 加载完成")
print(f"  模型参数量：{sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# 第 2 步：准备音频
# ============================================================================
print("\n" + "=" * 60)
print("【2. 准备测试音频】")
print("=" * 60)

# 方法 1：使用示例音频
print("使用内置示例音频...")

# 如果没有音频，创建一个简单的测试
# 实际使用时替换成你的音频文件
audio_path = "test_audio.wav"

try:
    # 加载音频
    audio = whisper.load_audio(audio_path)
    print(f"✓ 音频加载成功")
    print(f"  时长：{len(audio)/16000:.2f} 秒")
    print(f"  采样率：16000 Hz")
    
    # 可视化声波
    fig, ax = plt.subplots(figsize=(14, 4))
    time = np.arange(0, len(audio)) / 16000
    ax.plot(time, audio, linewidth=0.5)
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('振幅', fontsize=12)
    ax.set_title('声波图', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"无法加载音频：{e}")
    print("将使用模拟数据进行演示")
    
    # 创建模拟音频（正弦波组合）
    duration = 3  # 3 秒
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 组合几个频率（模拟元音）
    audio = (0.3 * np.sin(2 * np.pi * 300 * t) + 
             0.3 * np.sin(2 * np.pi * 500 * t) +
             0.3 * np.sin(2 * np.pi * 800 * t))
    
    # 加窗函数（让它像语音）
    window = np.hanning(len(audio))
    audio *= window
    
    print(f"✓ 已创建模拟音频")
    print(f"  时长：{duration} 秒")

# ============================================================================
# 第 3 步：语音识别
# ============================================================================
print("\n" + "=" * 60)
print("【3. 开始语音识别】")
print("=" * 60)

print("正在识别...")

# 设置选项
options = {
    "language": "zh",  # 中文
    "task": "transcribe",  # 转写（而不是翻译）
    "verbose": False
}

# 进行识别
result = model.transcribe(
    audio_path if 'audio_path' in locals() else audio,
    language=options["language"],
    task=options["task"]
)

text = result["text"]
segments = result["segments"]

print(f"\n✅ 识别完成！")
print(f"\n识别结果:")
print(f"  {text}")

# ============================================================================
# 第 4 步：显示详细结果
# ============================================================================
print("\n" + "=" * 60)
print("【4. 详细分析】")
print("=" * 60)

print(f"\n完整文本:")
print(f"  {text}")

print(f"\n分段结果:")
for i, segment in enumerate(segments, 1):
    start = segment["start"]
    end = segment["end"]
    content = segment["text"]
    
    print(f"\n片段 {i}:")
    print(f"  时间：{start:.2f}s - {end:.2f}s")
    print(f"  内容：{content}")
    print(f"  时长：{end-start:.2f}秒")

# ============================================================================
# 第 5 步：可视化注意力（如果可用）
# ============================================================================
if hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
    print("\n" + "=" * 60)
    print("📊 可视化 Attention 权重")
    print("=" * 60)
    
    try:
        # 获取最后一个 decoder layer 的 attention
        with torch.no_grad():
            # 这里简化演示，实际会更复杂
            print("注意：完整的 attention 可视化需要更复杂的处理")
            print("这里展示概念图")
        
        # 画一个概念图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 模拟 attention 矩阵
        n_frames = 50  # 音频帧数
        n_tokens = len(text) * 2  # 文字 token 数
        
        # 创建一个对角线为主的矩阵（模拟对齐）
        attention = np.zeros((n_tokens, n_frames))
        for i in range(n_tokens):
            center = int(i * n_frames / n_tokens)
            width = max(3, n_frames // (n_tokens * 2))
            start = max(0, center - width)
            end = min(n_frames, center + width)
            attention[i, start:end] = 1.0
        
        # 归一化
        attention = attention / attention.sum(axis=1, keepdims=True)
        
        im = ax.imshow(attention, cmap='Blues', aspect='auto')
        ax.set_xlabel('音频帧', fontsize=12)
        ax.set_ylabel('文字 Token', fontsize=12)
        ax.set_title('Attention 权重示意图', fontsize=14)
        plt.colorbar(im, ax=ax, label='Attention 强度')
        plt.tight_layout()
        plt.show()
        
        print("\n💡 Attention 的作用:")
        print("  - 识别每个字时，关注对应的音频部分")
        print("  - 亮色区域 = 高关注度")
        print("  - 可以看到字和音频的对齐关系")
        
    except Exception as e:
        print(f"Attention 可视化失败：{e}")
        print("这不影响使用，继续下面的内容")

print("\n🎊 恭喜！你完成了 Whisper 语音识别实战！")
print("=" * 60)

# ============================================================================
# 第 6 步：实际应用建议
# ============================================================================
print("\n" + "=" * 60)
print("【6. 实际应用建议】")
print("=" * 60)

print("""
使用场景推荐:

1. 会议记录:
   ✓ 录制会议音频
   ✓ 用 Whisper 转文字
   ✓ 人工校对关键信息
   ✓ 自动生成会议纪要

2. 视频字幕:
   ✓ 提取视频音频
   ✓ 语音识别生成字幕
   ✓ 调整时间轴
   ✓ 导出 SRT 格式

3. 语音输入法:
   ✓ 实时录音
   ✓ 流式识别
   ✓ 即时显示文字
   ✓ 支持标点符号

4. 客服质检:
   ✓ 录音转文字
   ✓ 关键词检测
   ✓ 情感分析
   ✓ 自动生成报告

注意事项:

✓ 音频质量很重要
  - 尽量清晰、无噪音
  - 采样率至少 16kHz
  
✓ 选择合适的模型
  - tiny/base: 快速测试
  - small/medium: 生产环境
  - large: 追求最佳效果

✓ 后处理有必要
  - 纠正同音词错误
  - 添加标点符号
  - 格式化数字、日期

✓ 隐私保护
  - 敏感数据本地处理
  - 不要上传机密信息
""")

print("\n🎉 语音识别实战完成！")
print("=" * 60)