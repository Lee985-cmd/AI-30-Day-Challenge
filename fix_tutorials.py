#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量修复 tutorials 目录下所有 Day README 的问题
"""

import os
from pathlib import Path

# 每天的主题名称
day_topics = {
    1: "Python 和 NumPy 基础",
    2: "监督学习实战（KNN）",
    3: "决策树和随机森林",
    4: "支持向量机 SVM",
    5: "K-means 聚类",
    6: "模型评估和优化",
    7: "Week1 复习和小项目",
    8: "神经网络初探",
    9: "多层神经网络",
    10: "PyTorch 入门",
    11: "CNN 基础",
    12: "经典 CNN 架构",
    13: "RNN 和 LSTM",
    14: "Week2 综合项目（CIFAR-10）",
    15: "目标检测基础（YOLO）",
    16: "YOLO 实时检测",
    17: "Faster R-CNN",
    18: "图像分割基础",
    19: "GAN 生成对抗网络",
    20: "语音识别基础",
    21: "Week3 综合项目",
    22: "Transformer 基础",
    23: "BERT 和大语言模型",
    24: "情感分析和 Prompt Engineering",
    25: "强化学习入门",
    26: "Flappy Bird AI 实战",
    27: "模型部署和工程化",
    28: "AI 伦理和安全",
    29: "前沿技术概览",
    30: "毕业项目和职业规划"
}

tutorials_dir = Path("tutorials")

def fix_readme(day_num):
    """修复单个 Day 的 README"""
    day_dir = tutorials_dir / f"Day{day_num:02d}"
    readme_file = day_dir / "README.md"
    
    if not readme_file.exists():
        print(f"❌ Day{day_num:02d}: README.md 不存在，跳过")
        return False
    
    content = readme_file.read_text(encoding='utf-8')
    original_content = content
    topic = day_topics.get(day_num, "未知主题")
    
    # 1. 替换标题中的占位符
    if "[主题名称]" in content:
        content = content.replace(f"# Day{day_num:02d} - [主题名称]", 
                                  f"# Day{day_num:02d} - {topic}")
    
    # 2. 替换 Q1-Q6 的占位符
    for q_num in range(1, 7):
        content = content.replace(f"#### Q{q_num} - [主题{q_num}]", 
                                  f"#### Q{q_num}")
    
    # 3. 修复不存在的链接（Q0 和 Q6 可能不存在）
    # 检查实际存在的文件
    md_files = [f.name for f in day_dir.glob("*.md")]
    
    # 检查 Q0 是否存在
    q0_exists = any("Q0" in f for f in md_files)
    if not q0_exists:
        # 删除 Q0 相关部分
        import re
        content = re.sub(r'### Q0 - 快速复习\n\[\ud83d\udcd6 Day\d{2}-Q0\]\(\./Day\d{2}-Q0\*\.md\)\n\n.*?预计时间:.*?\n\n---\n\n', 
                        '', content, flags=re.DOTALL)
        content = content.replace("### Q0 - 快速复习\n[📖 Day{:02d}-Q0](./Day{:02d}-Q0*.md)\n\n".format(day_num, day_num), "")
    
    # 检查 Q6 是否存在
    q6_exists = any("Q6" in f for f in md_files)
    if not q6_exists:
        # 删除 Q6 相关部分
        content = content.replace(f"#### Q6 - [主题6/进阶]\n[💡 Day{day_num:02d}-Q6](./Day{day_num:02d}-Q6*.md)\n\n", "")
        content = content.replace(f"#### Q6 - [主题6]\n[💡 Day{day_num:02d}-Q6](./Day{day_num:02d}-Q6*.md)\n\n", "")
    
    # 4. 修复前后链接
    prev_day = day_num - 1 if day_num > 1 else None
    next_day = day_num + 1 if day_num < 30 else None
    
    if prev_day:
        content = content.replace(f"[← 前一天](../Day{prev_day:02d}/README.md)", 
                                  f"[← Day{prev_day:02d}: {day_topics[prev_day]}](../Day{prev_day:02d}/README.md)")
    else:
        content = content.replace("[← 前一天](../Day00/README.md)", "[← 回到首页](README.md)")
    
    if next_day:
        content = content.replace(f"[→ 后一天](../Day{next_day:02d}/README.md)", 
                                  f"[→ Day{next_day:02d}: {day_topics[next_day]}](../Day{next_day:02d}/README.md)")
    else:
        # Day30 没有后一天
        content = content.replace("[→ 后一天](../Day31/README.md)", "[→ 回到首页](README.md)")
    
    # 5. 修复 Week 相关链接中的无效天数
    if day_num in [29, 30]:
        # 删除无效的 Day31-35 链接
        import re
        content = re.sub(r'\n- \[Day3[1-5]: 主题\]\(Day3[1-5]/README\.md\)', '', content)
        content = re.sub(r'\n- ⏳ Day3[1-5]', '', content)
    
    # 6. 更新链接（如果链接不存在就删除）
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if './Day' in line and 'md' in line and '*' in line:
            # 检查这个链接是否指向真实文件
            import re
            import glob
            matches = re.findall(r'\./([^)]+\.md)', line)
            if matches:
                match = matches[0]
                full_pattern = str(day_dir / match)
                matched_files = glob.glob(full_pattern)
                if not matched_files:
                    # 链接不存在，删除这行
                    continue
        new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    # 保存修复后的文件
    readme_file.write_text(content, encoding='utf-8')
    
    if content != original_content:
        print(f"✅ Day{day_num:02d}: 已修复")
        return True
    else:
        print(f"⚪ Day{day_num:02d}: 无需修复")
        return False

# 批量修复所有 Day
print("=" * 80)
print("🔧 开始批量修复 tutorials/ 目录下的 README 文件")
print("=" * 80)

fixed_count = 0
for day_num in range(1, 31):
    if fix_readme(day_num):
        fixed_count += 1

print("\n" + "=" * 80)
print(f"✅ 修复完成！共修复了 {fixed_count} 个 README 文件")
print("=" * 80)
