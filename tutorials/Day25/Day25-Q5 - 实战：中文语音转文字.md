# Day25-Q5 - 实战：中文语音转文字

## 🎯 项目目标

构建一个完整的中文语音转文字系统，支持批量处理、字幕生成和会议记录。

## 📦 完整项目代码

### 项目结构

```
chinese_asr_project/
├── main.py              # 主程序
├── transcriber.py       # 转写器类
├── utils.py             # 工具函数
├── config.py            # 配置文件
├── audio/               # 输入音频
├── output/              # 输出结果
└── requirements.txt     # 依赖
```

### 1. 配置文件 (config.py)

```python
"""配置文件"""

# 模型设置
MODEL_SIZE = "small"  # tiny, base, small, medium, large
DEVICE = "cuda"       # cuda 或 cpu

# 语言设置
LANGUAGE = "zh"       # zh (中文), en (英文), 等
TASK = "transcribe"   # transcribe 或 translate

# 解码参数
TEMPERATURE = 0.0
BEAM_SIZE = 5
BEST_OF = 5

# 输出设置
OUTPUT_DIR = "./output"
SAVE_JSON = True
SAVE_TXT = True
SAVE_SRT = True

# 音频设置
SAMPLE_RATE = 16000
```

### 2. 转写器类 (transcriber.py)

```python
import whisper
import torch
from pathlib import Path
import json
from datetime import datetime

class ChineseTranscriber:
    """中文语音转写器"""
    
    def __init__(self, model_size="small", device=None):
        """
        初始化转写器
        
        参数:
        model_size: 模型大小 (tiny/base/small/medium/large)
        device: 设备 (cuda/cpu)
        """
        
        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"使用设备: {self.device}")
        
        # 加载模型
        print(f"加载 Whisper {model_size} 模型...")
        self.model = whisper.load_model(model_size)
        self.model = self.model.to(self.device)
        print("✓ 模型加载完成\n")
        
        self.model_size = model_size
    
    def transcribe_file(self, audio_path, language='zh', **kwargs):
        """
        转写单个音频文件
        
        参数:
        audio_path: 音频文件路径
        language: 语言代码
        **kwargs: 其他参数
        
        返回:
        result: 转写结果字典
        """
        
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"文件不存在: {audio_path}")
        
        print(f"处理: {audio_path.name}")
        
        # 转写
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            task='transcribe',
            verbose=False,
            **kwargs
        )
        
        print(f"  ✓ 完成 (时长: {result['segments'][-1]['end']:.1f}秒)\n")
        
        return result
    
    def batch_transcribe(self, input_dir, output_dir=None, language='zh'):
        """
        批量转写目录中的所有音频
        
        参数:
        input_dir: 输入目录
        output_dir: 输出目录
        language: 语言代码
        
        返回:
        results: 所有结果的字典
        """
        
        input_path = Path(input_dir)
        
        if output_dir is None:
            output_path = input_path / "output"
        else:
            output_path = Path(output_dir)
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取音频文件
        audio_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        audio_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in audio_extensions
        ]
        
        if not audio_files:
            print(f"在 {input_dir} 中未找到音频文件")
            return {}
        
        print(f"找到 {len(audio_files)} 个音频文件\n")
        print("=" * 60)
        
        results = {}
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}]", end=" ")
            
            try:
                result = self.transcribe_file(audio_file, language)
                results[audio_file.stem] = result
                
                # 保存结果
                self.save_results(result, output_path, audio_file.stem)
                
            except Exception as e:
                print(f"  ✗ 错误: {e}\n")
                results[audio_file.stem] = {'error': str(e)}
        
        print("=" * 60)
        print(f"\n批量处理完成! 成功: {len(results)} 个文件")
        print(f"输出目录: {output_path}\n")
        
        return results
    
    def save_results(self, result, output_dir, filename):
        """
        保存转写结果
        
        参数:
        result: 转写结果
        output_dir: 输出目录
        filename: 文件名 (不含扩展名)
        """
        
        output_path = Path(output_dir)
        
        # 保存 TXT
        txt_path = output_path / f"{filename}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        # 保存 JSON
        json_path = output_path / f"{filename}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        # 保存 SRT
        srt_path = output_path / f"{filename}.srt"
        self.save_as_srt(result, srt_path)
        
        # 保存 Markdown
        md_path = output_path / f"{filename}.md"
        self.save_as_markdown(result, md_path)
    
    def save_as_srt(self, result, output_path):
        """保存为 SRT 字幕格式"""
        
        def format_timestamp(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millisecs = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start = format_timestamp(segment['start'])
                end = format_timestamp(segment['end'])
                text = segment['text'].strip()
                
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{text}\n\n")
    
    def save_as_markdown(self, result, output_path):
        """保存为 Markdown 格式"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# 语音转写结果\n\n")
            f.write(f"**文件:** {Path(output_path).stem}\n")
            f.write(f"**语言:** {result.get('language', 'unknown')}\n")
            
            if result['segments']:
                duration = result['segments'][-1]['end']
                f.write(f"**时长:** {duration:.0f} 秒 ({duration/60:.1f} 分钟)\n")
            
            f.write(f"**模型:** Whisper {self.model_size}\n")
            f.write(f"**时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            for i, segment in enumerate(result['segments'], 1):
                start_min = int(segment['start'] // 60)
                start_sec = int(segment['start'] % 60)
                
                f.write(f"### [{start_min:02d}:{start_sec:02d}]\n\n")
                f.write(f"{segment['text'].strip()}\n\n")
    
    def get_statistics(self, results):
        """
        获取统计信息
        
        参数:
        results: 转写结果字典
        
        返回:
        stats: 统计信息字典
        """
        
        total_duration = 0
        total_segments = 0
        total_words = 0
        
        for name, result in results.items():
            if 'error' in result:
                continue
            
            if result['segments']:
                total_duration += result['segments'][-1]['end']
                total_segments += len(result['segments'])
                total_words += len(result['text'].split())
        
        stats = {
            'total_files': len(results),
            'successful': sum(1 for r in results.values() if 'error' not in r),
            'failed': sum(1 for r in results.values() if 'error' in r),
            'total_duration': total_duration,
            'total_segments': total_segments,
            'total_words': total_words,
        }
        
        return stats
```

### 3. 工具函数 (utils.py)

```python
import os
import subprocess
from pathlib import Path

def check_ffmpeg():
    """检查 FFmpeg 是否安装"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def convert_audio_format(input_file, output_file=None, format='wav'):
    """
    转换音频格式
    
    参数:
    input_file: 输入文件
    output_file: 输出文件 (可选)
    format: 输出格式 (wav/mp3/flac)
    
    返回:
    output_file: 输出文件路径
    """
    
    if output_file is None:
        output_file = Path(input_file).with_suffix(f'.{format}')
    
    cmd = [
        'ffmpeg', '-i', str(input_file),
        '-ar', '16000',  # 采样率 16kHz
        '-ac', '1',      # 单声道
        '-y',            # 覆盖已存在文件
        str(output_file)
    ]
    
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    return output_file

def get_audio_info(audio_file):
    """
    获取音频文件信息
    
    返回:
    info: 音频信息字典
    """
    
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        str(audio_file)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    import json
    info = json.loads(result.stdout)
    
    return {
        'duration': float(info['format']['duration']),
        'size': int(info['format']['size']),
        'bit_rate': int(info['format'].get('bit_rate', 0)),
    }

def split_long_audio(audio_file, chunk_duration=30, output_dir=None):
    """
    分割长音频文件
    
    Whisper 最长支持 30 秒
    超过需要分割
    
    参数:
    audio_file: 音频文件
    chunk_duration: 每段时长 (秒)
    output_dir: 输出目录
    
    返回:
    chunks: 分割后的文件列表
    """
    
    if output_dir is None:
        output_dir = Path(audio_file).parent / "chunks"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取总时长
    info = get_audio_info(audio_file)
    total_duration = info['duration']
    
    if total_duration <= chunk_duration:
        return [audio_file]
    
    # 分割
    chunks = []
    num_chunks = int(total_duration / chunk_duration) + 1
    
    for i in range(num_chunks):
        start = i * chunk_duration
        output_file = output_dir / f"{Path(audio_file).stem}_chunk_{i:03d}.wav"
        
        cmd = [
            'ffmpeg', '-i', str(audio_file),
            '-ss', str(start),
            '-t', str(chunk_duration),
            '-ar', '16000',
            '-ac', '1',
            '-y',
            str(output_file)
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if output_file.exists():
            chunks.append(output_file)
    
    print(f"分割完成: {len(chunks)} 个片段")
    
    return chunks
```

### 4. 主程序 (main.py)

```python
#!/usr/bin/env python3
"""
中文语音转文字系统
基于 OpenAI Whisper
"""

import argparse
from pathlib import Path
from transcriber import ChineseTranscriber
from utils import check_ffmpeg, get_audio_info

def main():
    parser = argparse.ArgumentParser(
        description='中文语音转文字系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转写单个文件
  python main.py audio.wav
  
  # 批量转写目录
  python main.py ./audio_folder
  
  # 指定模型大小
  python main.py audio.wav --model large
  
  # 指定输出目录
  python main.py audio.wav --output ./results
        """
    )
    
    parser.add_argument(
        'input',
        help='输入音频文件或目录'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='small',
        help='模型大小 (默认: small)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='输出目录 (默认: 输入目录/output)'
    )
    
    parser.add_argument(
        '--language', '-l',
        default='zh',
        help='语言代码 (默认: zh)'
    )
    
    parser.add_argument(
        '--device', '-d',
        choices=['cuda', 'cpu'],
        default=None,
        help='计算设备 (默认: 自动检测)'
    )
    
    args = parser.parse_args()
    
    # 检查 FFmpeg
    if not check_ffmpeg():
        print("❌ 错误: 未找到 FFmpeg")
        print("请安装 FFmpeg: https://ffmpeg.org/download.html")
        return
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"❌ 错误: 文件或目录不存在: {args.input}")
        return
    
    # 创建转写器
    transcriber = ChineseTranscriber(
        model_size=args.model,
        device=args.device
    )
    
    # 执行转写
    if input_path.is_file():
        # 单个文件
        print(f"\n开始转写: {input_path.name}\n")
        result = transcriber.transcribe_file(
            input_path,
            language=args.language
        )
        
        # 显示结果
        print("\n" + "=" * 60)
        print("转写结果:")
        print("=" * 60)
        print(result['text'])
        print("=" * 60)
        
        # 保存
        output_dir = Path(args.output) if args.output else input_path.parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        transcriber.save_results(result, output_dir, input_path.stem)
        
        print(f"\n✓ 结果已保存到: {output_dir}\n")
    
    elif input_path.is_dir():
        # 批量处理
        print(f"\n开始批量转写: {input_path}\n")
        results = transcriber.batch_transcribe(
            input_path,
            output_dir=args.output,
            language=args.language
        )
        
        # 统计信息
        if results:
            stats = transcriber.get_statistics(results)
            print("\n统计信息:")
            print(f"  总文件数: {stats['total_files']}")
            print(f"  成功: {stats['successful']}")
            print(f"  失败: {stats['failed']}")
            print(f"  总时长: {stats['total_duration']:.0f} 秒 ({stats['total_duration']/60:.1f} 分钟)")
            print(f"  总片段: {stats['total_segments']}")
            print(f"  总词数: {stats['total_words']}")

if __name__ == '__main__':
    main()
```

### 5. 依赖文件 (requirements.txt)

```
openai-whisper>=20231117
torch>=2.0.0
torchaudio>=2.0.0
numpy>=1.24.0
librosa>=0.10.0
sounddevice>=0.4.6
```

## 🚀 使用方法

### 1. 安装

```bash
# 克隆或下载项目
cd chinese_asr_project

# 安装依赖
pip install -r requirements.txt

# 安装 FFmpeg
# Windows: 下载并添加到 PATH
# Mac: brew install ffmpeg
# Linux: sudo apt-get install ffmpeg
```

### 2. 基本使用

```bash
# 转写单个文件
python main.py audio.wav

# 批量转写
python main.py ./audio_folder

# 使用大模型
python main.py audio.wav --model large

# 指定输出目录
python main.py audio.wav --output ./results

# 使用 CPU
python main.py audio.wav --device cpu
```

### 3. Python API 使用

```python
from transcriber import ChineseTranscriber

# 创建转写器
transcriber = ChineseTranscriber(model_size="small")

# 转写单个文件
result = transcriber.transcribe_file("meeting.wav")
print(result['text'])

# 批量转写
results = transcriber.batch_transcribe("./meetings", "./output")

# 获取统计
stats = transcriber.get_statistics(results)
print(f"处理了 {stats['successful']} 个文件")
```

## 📊 输出示例

### TXT 格式
```
今天我们来讨论人工智能的发展。近年来，深度学习技术取得了巨大进步。
特别是在自然语言处理领域，Transformer 架构带来了革命性的变化。
```

### SRT 格式
```
1
00:00:01,000 --> 00:00:05,000
今天我们来讨论人工智能的发展。

2
00:00:05,500 --> 00:00:10,000
近年来，深度学习技术取得了巨大进步。
```

### Markdown 格式
```markdown
# 语音转写结果

**文件:** meeting
**语言:** zh
**时长:** 120 秒 (2.0 分钟)
**模型:** Whisper small
**时间:** 2024-01-18 14:30:00

---

### [00:00]

今天我们来讨论人工智能的发展。

### [00:05]

近年来，深度学习技术取得了巨大进步。
```

## 💡 优化建议

### 1. 提高准确率

```python
# 使用更大的模型
transcriber = ChineseTranscriber(model_size="large")

# 提供上下文提示
result = model.transcribe(
    "audio.wav",
    initial_prompt="这是一段关于人工智能的技术讲座"
)

# 微调模型
# 针对特定领域训练
```

### 2. 提高速度

```python
# 使用更小的模型
transcriber = ChineseTranscriber(model_size="tiny")

# GPU 加速
transcriber = ChineseTranscriber(device="cuda")

# 并行处理
from multiprocessing import Pool
```

### 3. 处理长音频

```python
from utils import split_long_audio

# 分割长音频
chunks = split_long_audio("long_meeting.wav", chunk_duration=30)

# 分别转写
for chunk in chunks:
    result = transcriber.transcribe_file(chunk)
    # 合并结果
```

## 🎓 项目总结

通过这个实战项目，我们学会了：

1. **Whisper 的使用**
   - 加载模型
   - 转写音频
   - 参数调优

2. **批量处理**
   - 目录扫描
   - 并行处理
   - 结果保存

3. **多种格式输出**
   - TXT
   - JSON
   - SRT
   - Markdown

4. **工程化实践**
   - 模块化设计
   - 配置管理
   - 错误处理

## 🚀 下一步

现在我们已经完成了中文语音转文字的实战项目，接下来让我们了解常见问题和解决方案。

---

**下一步：** [Day25-Q6 - 常见问题和最佳实践](./Day25-Q6%20-%20常见问题和最佳实践.md)