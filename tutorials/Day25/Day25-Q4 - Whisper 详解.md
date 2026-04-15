# Day25-Q4 - Whisper 详解与实战

## 🎤 Whisper 完整实战指南

### 安装和设置

```bash
# 安装 Whisper
pip install openai-whisper

# 安装 FFmpeg (必需)
# Windows: 下载 https://ffmpeg.org/download.html
# Mac: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg

# 可选：GPU 加速
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 基础识别

```python
import whisper

# 加载模型
model = whisper.load_model("base")

# 识别音频
result = model.transcribe("audio.wav")

# 输出结果
print(result["text"])
```

**支持的音频格式:**
- WAV
- MP3
- M4A
- FLAC
- OGG
- 等 (FFmpeg 支持的所有格式)

### 高级选项

```python
result = model.transcribe(
    "audio.mp3",
    
    # 语言设置
    language="zh",           # 指定语言
    task="transcribe",       # transcribe 或 translate
    
    # 温度控制 (提高鲁棒性)
    temperature=0.0,         # 单一温度
    # 或自动调整
    temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    
    # 解码策略
    beam_size=5,             # 束搜索大小
    best_of=5,               # 采样次数
    
    # 惩罚参数
    patience=1.0,            # 束搜索耐心度
    length_penalty=1.0,      # 长度惩罚
    repetition_penalty=1.0,  # 重复惩罚
    
    # 其他
    no_speech_threshold=0.6, # 无语音阈值
    logprob_threshold=-1.0,  # 对数概率阈值
)
```

## 🔧 实用功能

### 1. 带时间戳的转写

```python
result = model.transcribe("audio.wav", word_timestamps=True)

# 访问片段
for segment in result["segments"]:
    print(f"[{segment['start']:.2f} - {segment['end']:.2f}]")
    print(f"  {segment['text']}")
    print()

# 访问单词级时间戳
for segment in result["segments"]:
    for word in segment.get("words", []):
        print(f"[{word['start']:.2f} - {word['end']:.2f}] {word['word']}")
```

### 2. 语言检测

```python
# 自动检测语言
result = model.transcribe("unknown_language.wav")
print(f"检测到的语言: {result['language']}")
print(f"语言置信度: {result.get('language_probability', 'N/A')}")

# 支持的语言
print("\nWhisper 支持的语言:")
languages = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", 
    "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn",
    "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my",
    "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue"
]
print(f"共 {len(languages)} 种语言")
```

### 3. 翻译功能

```python
# 翻译成英文
result = model.transcribe("chinese_audio.wav", task="translate")
print(f"原文 (中文): ...")
print(f"译文 (英文): {result['text']}")

# 注意: Whisper 只能翻译成英文
# 如需其他语言，需要额外的翻译模型
```

### 4. 批量处理

```python
from pathlib import Path
import json

def batch_transcribe_directory(input_dir, output_dir, model_size="base"):
    """批量转写目录中的所有音频"""
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    model = whisper.load_model(model_size)
    
    # 获取所有音频文件
    audio_files = list(Path(input_dir).glob("*"))
    audio_files = [f for f in audio_files if f.suffix.lower() in 
                   ['.wav', '.mp3', '.m4a', '.flac', '.ogg']]
    
    print(f"找到 {len(audio_files)} 个音频文件\n")
    
    results = {}
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] 处理: {audio_file.name}")
        
        try:
            # 转写
            result = model.transcribe(str(audio_file))
            
            # 保存文本
            txt_path = Path(output_dir) / f"{audio_file.stem}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            
            # 保存 JSON (包含详细信息)
            json_path = Path(output_dir) / f"{audio_file.stem}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            results[audio_file.stem] = result['text']
            print(f"  ✓ 完成\n")
        
        except Exception as e:
            print(f"  ✗ 错误: {e}\n")
    
    return results

# 使用
results = batch_transcribe_directory(
    "./audio_input",
    "./transcription_output",
    model_size="small"
)
```

## 🚀 性能优化

### GPU 加速

```python
import torch

# 检查 CUDA
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# 加载模型到 GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base")
model = model.to(device)

# 性能对比
import time

audio_file = "test.wav"

# CPU
model_cpu = whisper.load_model("base").to("cpu")
start = time.time()
result_cpu = model_cpu.transcribe(audio_file)
cpu_time = time.time() - start
print(f"CPU 耗时: {cpu_time:.2f} 秒")

# GPU
model_gpu = whisper.load_model("base").to("cuda")
start = time.time()
result_gpu = model_gpu.transcribe(audio_file)
gpu_time = time.time() - start
print(f"GPU 耗时: {gpu_time:.2f} 秒")
print(f"加速比: {cpu_time / gpu_time:.1f}x")

# 典型加速比:
# base: 5-10x
# small: 8-15x
# medium: 10-20x
# large: 15-25x
```

### 模型量化

```python
# 使用更小的模型
models = {
    "tiny": "最快，准确率较低",
    "base": "快速，适合一般用途",
    "small": "平衡性能和速度",
    "medium": "高准确率",
    "large": "最高准确率，最慢"
}

for name, desc in models.items():
    print(f"{name:8s}: {desc}")

# 推荐:
# - 实时应用: tiny 或 base
# - 一般用途: small
# - 高质量: medium 或 large
```

### 批处理优化

```python
# Whisper 本身不支持真正的批处理
# 但可以并行处理多个文件

from concurrent.futures import ThreadPoolExecutor
import threading

def transcribe_single(model, audio_file):
    """单个文件转写"""
    result = model.transcribe(str(audio_file))
    return audio_file.stem, result['text']

def parallel_transcribe(audio_files, model_size="base", max_workers=4):
    """并行转写多个文件"""
    
    # 每个线程加载自己的模型 (避免冲突)
    def worker(audio_file):
        model = whisper.load_model(model_size)
        return transcribe_single(model, audio_file)
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, f): f for f in audio_files}
        
        for future in futures:
            audio_file = futures[future]
            try:
                name, text = future.result()
                results[name] = text
                print(f"✓ {audio_file.name}")
            except Exception as e:
                print(f"✗ {audio_file.name}: {e}")
    
    return results

# 使用
from pathlib import Path
audio_files = list(Path("./audio").glob("*.wav"))
results = parallel_transcribe(audio_files, model_size="base", max_workers=4)
```

## 📊 实际应用案例

### 案例 1: 会议记录转写

```python
class MeetingTranscriber:
    """会议记录转写器"""
    
    def __init__(self, model_size="small"):
        self.model = whisper.load_model(model_size)
    
    def transcribe_meeting(self, audio_file, speakers=None):
        """
        转写会议录音
        
        参数:
        audio_file: 音频文件路径
        speakers: 说话人列表 (可选)
        
        返回:
        结构化的会议记录
        """
        
        # 转写
        result = self.model.transcribe(
            audio_file,
            word_timestamps=True,
            language='zh'
        )
        
        # 格式化输出
        transcript = {
            'title': Path(audio_file).stem,
            'language': result['language'],
            'duration': result.get('segments', [{}])[-1].get('end', 0) if result['segments'] else 0,
            'segments': []
        }
        
        for segment in result['segments']:
            transcript['segments'].append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'speaker': None  # 需要说话人分离
            })
        
        return transcript
    
    def save_as_markdown(self, transcript, output_file):
        """保存为 Markdown 格式"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 会议记录\n\n")
            f.write(f"**文件:** {transcript['title']}\n")
            f.write(f"**语言:** {transcript['language']}\n")
            f.write(f"**时长:** {transcript['duration']:.0f} 秒\n\n")
            f.write("---\n\n")
            
            for i, seg in enumerate(transcript['segments'], 1):
                start_min = int(seg['start'] // 60)
                start_sec = int(seg['start'] % 60)
                
                f.write(f"### [{start_min:02d}:{start_sec:02d}]\n\n")
                f.write(f"{seg['text']}\n\n")
        
        print(f"✓ 保存到: {output_file}")

# 使用
transcriber = MeetingTranscriber(model_size="small")
transcript = transcriber.transcribe_meeting("meeting.wav")
transcriber.save_as_markdown(transcript, "meeting_notes.md")
```

### 案例 2: 视频字幕生成

```python
def generate_srt(audio_file, output_srt, model_size="base"):
    """
    生成 SRT 字幕文件
    
    SRT 格式:
    1
    00:00:01,000 --> 00:00:04,000
    第一句字幕
    
    2
    00:00:05,000 --> 00:00:08,000
    第二句字幕
    """
    
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_file, word_timestamps=True)
    
    def format_timestamp(seconds):
        """格式化时间戳为 SRT 格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    with open(output_srt, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result['segments'], 1):
            start = format_timestamp(segment['start'])
            end = format_timestamp(segment['end'])
            text = segment['text'].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{text}\n\n")
    
    print(f"✓ SRT 字幕已生成: {output_srt}")

# 使用
generate_srt("video_audio.wav", "subtitles.srt", model_size="small")
```

### 案例 3: 实时转录 (模拟)

```python
import sounddevice as sd
import numpy as np
import queue

class RealTimeTranscriber:
    """实时转录器 (简化版)"""
    
    def __init__(self, model_size="tiny"):
        self.model = whisper.load_model(model_size)
        self.audio_queue = queue.Queue()
        self.sample_rate = 16000
        self.chunk_duration = 5  # 每 5 秒处理一次
    
    def audio_callback(self, indata, frames, time, status):
        """音频回调"""
        if status:
            print(f"状态: {status}")
        self.audio_queue.put(indata.copy())
    
    def transcribe_chunk(self, audio_data):
        """转写音频块"""
        # 保存为临时文件
        import scipy.io.wavfile as wavfile
        temp_file = "temp_chunk.wav"
        wavfile.write(temp_file, self.sample_rate, audio_data)
        
        # 转写
        result = self.model.transcribe(temp_file, language='zh')
        
        return result['text']
    
    def start(self, duration=60):
        """开始实时转录"""
        print("开始实时转录... (按 Ctrl+C 停止)")
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=self.audio_callback
            ):
                # 收集音频
                chunks = []
                start_time = sd.get_stream().time
                
                while sd.get_stream().time - start_time < duration:
                    sd.sleep(100)  # 等待 100ms
                    
                    # 定期处理
                    if not self.audio_queue.empty():
                        chunk = self.audio_queue.get()
                        chunks.append(chunk)
                        
                        # 每 5 秒处理一次
                        if len(chunks) >= int(self.chunk_duration * 10):
                            audio_data = np.concatenate(chunks)
                            text = self.transcribe_chunk(audio_data)
                            print(f"\n{text}\n")
                            chunks = []  # 清空
            
        except KeyboardInterrupt:
            print("\n停止转录")

# 注意: 真正的实时转录需要更复杂的实现
# 这里只是演示概念
# transcriber = RealTimeTranscriber(model_size="tiny")
# transcriber.start(duration=30)
```

## 🛠️ 微调和定制

### 微调 Whisper

```python
"""
何时需要微调?

✓ 专业领域 (医学、法律、技术)
✓ 特定口音或方言
✓ 专有名词很多
✓ 标准模型效果不佳

微调步骤:

1. 准备数据集
   - 音频文件
   - 对应的文本标注
   
2. 选择基础模型
   - 通常从 small 或 base 开始
   
3. 训练
   - 学习率: 1e-5
   - 批次大小: 根据显存
   - 轮数: 3-10 epochs
   
4. 评估
   - 在验证集上测试 WER
   
5. 部署
   - 保存模型
   - 集成到应用

工具:
- Hugging Face Transformers
- NVIDIA NeMo
- OpenAI Whisper (官方)
"""

# 示例: 使用 Hugging Face 微调
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# 加载预训练模型
processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# 加载数据集
dataset = load_dataset("common_voice", "zh-CN")

# 预处理
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    batch["labels"] = processor(
        text=batch["sentence"]
    ).input_ids
    
    return batch

dataset = dataset.map(prepare_dataset)

# 训练配置
from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 训练
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()

# 保存
model.save_pretrained("./whisper-chinese-custom")
processor.save_pretrained("./whisper-chinese-custom")
```

### 词汇表扩展

```python
"""
添加专业术语:

方法 1: 后处理替换
"""

def replace_special_terms(text):
    """替换专业术语"""
    
    replacements = {
        "阿尔法": "α",
        "贝塔": "β",
        "伽马": "γ",
        "API": "API",
        "CEO": "CEO",
        # 添加更多...
    }
    
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)
    
    return text

# 使用
result = model.transcribe("technical_talk.wav")
corrected_text = replace_special_terms(result['text'])

"""
方法 2: 提示工程
"""

prompt = "以下是关于人工智能的技术讲座，包含很多专业术语如 Transformer、BERT、GPT 等。"

result = model.transcribe(
    "ai_lecture.wav",
    initial_prompt=prompt  # 提供上下文提示
)
```

## 💡 最佳实践

### 1. 模型选择

```
场景推荐:

实时字幕:
→ tiny 或 base
→ 延迟 < 1 秒

会议记录:
→ small 或 medium
→ 平衡速度和准确率

专业转写:
→ large
→ 最高准确率

多语言混合:
→ large-v2
→ 最好的语言切换
```

### 2. 音频预处理

```python
import librosa

def preprocess_audio(audio_file, target_sr=16000):
    """音频预处理"""
    
    # 加载音频
    signal, sr = librosa.load(audio_file, sr=target_sr)
    
    # 降噪 (简单版)
    # 可以使用更复杂的降噪算法
    
    # 归一化
    signal = signal / np.max(np.abs(signal))
    
    # 去除静音
    signal, _ = librosa.effects.trim(signal, top_db=20)
    
    return signal, target_sr

# 使用
signal, sr = preprocess_audio("noisy_audio.wav")
```

### 3. 后处理

```python
import re

def postprocess_text(text):
    """文本后处理"""
    
    # 修复常见错误
    text = text.replace("  ", " ")  # 多余空格
    text = re.sub(r'([。，！？])\1+', r'\1', text)  # 重复标点
    
    # 修正标点
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    
    # 首字母大写 (英文)
    # text = text.capitalize()
    
    return text.strip()

# 使用
result = model.transcribe("audio.wav")
clean_text = postprocess_text(result['text'])
```

### 4. 质量控制

```python
def quality_check(result, threshold=0.6):
    """质量检查"""
    
    issues = []
    
    # 检查置信度
    for segment in result['segments']:
        avg_logprob = segment.get('avg_logprob', 0)
        if avg_logprob < threshold:
            issues.append(f"低置信度片段: {segment['text'][:50]}...")
    
    # 检查异常长度
    for segment in result['segments']:
        duration = segment['end'] - segment['start']
        text_len = len(segment['text'])
        
        if duration > 0 and text_len / duration < 1:
            issues.append(f"可能漏识别: [{segment['start']:.1f}-{segment['end']:.1f}]")
    
    if issues:
        print("⚠️ 质量问题:")
        for issue in issues[:5]:  # 只显示前 5 个
            print(f"  - {issue}")
        return False
    else:
        print("✓ 质量检查通过")
        return True

# 使用
result = model.transcribe("audio.wav")
quality_check(result)
```

## 🎓 总结

### Whisper 的核心优势

1. **易用性** ⭐⭐⭐⭐⭐
   - 几行代码即可使用
   - 无需训练

2. **多语言** ⭐⭐⭐⭐⭐
   - 99 种语言
   - 自动检测

3. **鲁棒性** ⭐⭐⭐⭐
   - 噪声环境
   - 各种口音

4. **开源免费** ⭐⭐⭐⭐⭐
   - MIT 许可
   - 商业可用

### 局限性和应对

1. **延迟较高**
   → 使用小模型
   → GPU 加速

2. **专业术语**
   → 微调
   → 后处理

3. **标点不准确**
   → 专门的标点模型
   → 人工校对

## 🚀 下一步

现在我们已经掌握了 Whisper 的使用，接下来让我们通过完整的实战项目来巩固知识。

---

**下一步：** [Day25-Q5 - 实战：中文语音转文字](./Day25-Q5%20-%20实战：中文语音转文字.md)
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

![公众号二维码](../../images/logos/ewm.jpg)

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
