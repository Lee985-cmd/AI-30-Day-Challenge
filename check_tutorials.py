#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 tutorials 目录下所有 Day 的 README 质量
"""

import os
from pathlib import Path

tutorials_dir = Path("tutorials")

print("=" * 80)
print("🔍 检查 tutorials/ 目录下所有 Day README 质量")
print("=" * 80)

issues = []

for day_num in range(1, 31):
    day_dir = tutorials_dir / f"Day{day_num:02d}"
    readme_file = day_dir / "README.md"
    
    if not readme_file.exists():
        issues.append(f"❌ Day{day_num:02d}: README.md 不存在")
        continue
    
    content = readme_file.read_text(encoding='utf-8')
    lines = content.split('\n')
    
    # 检查问题
    day_issues = []
    
    # 1. 检查是否有占位符
    if '[主题名称]' in content or '[主题' in content:
        day_issues.append("含有占位符（[主题名称]、[主题1]等）")
    
    # 2. 检查行数（太短或太长都不好）
    if len(lines) < 20:
        day_issues.append(f"内容太短（{len(lines)} 行）")
    elif len(lines) > 100:
        day_issues.append(f"内容太长（{len(lines)} 行），可能冗余")
    
    # 3. 检查是否有有效的文件链接
    if 'README.md' not in content:
        day_issues.append("没有链接到其他文档")
    
    # 4. 检查日期
    if '2026年' in content:
        day_issues.append("日期是 2026 年（可能是模板）")
    
    # 5. 检查是否有学习目标
    if '学习目标' not in content and '学习目标' not in content:
        day_issues.append("缺少学习目标部分")
    
    # 6. 检查链接是否有效
    md_files = list(day_dir.glob("*.md"))
    for line in lines:
        if './Day' in line and 'md' in line:
            # 提取文件名
            import re
            matches = re.findall(r'\./([^)]+\.md)', line)
            for match in matches:
                if '*' in match:
                    # 通配符链接，检查是否真的匹配文件
                    import glob
                    full_pattern = str(day_dir / match)
                    matched_files = glob.glob(full_pattern)
                    if not matched_files:
                        day_issues.append(f"链接不存在: {match}")
    
    if day_issues:
        issues.append(f"\n⚠️  Day{day_num:02d}:")
        for issue in day_issues:
            issues.append(f"   - {issue}")
    else:
        print(f"✅ Day{day_num:02d}: 良好（{len(lines)} 行）")

print("\n" + "=" * 80)
print("📊 检查结果汇总")
print("=" * 80)

if issues:
    for issue in issues:
        print(issue)
else:
    print("🎉 所有 Day 的 README 都没有问题！")

print("\n" + "=" * 80)
