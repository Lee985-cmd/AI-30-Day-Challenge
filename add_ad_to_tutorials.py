"""
批量在教程末尾添加公众号引流文案
"""
import os
from pathlib import Path

# 引流文案内容（与 promotion-articles/公众号引流文案-教程通用版.md 一致）
ADDITIONAL_CONTENT = """
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
"""


def add_ad_to_markdown_files(root_dir):
    """在所有 DayXX 文件夹中的 md 文件末尾添加引流文案"""
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"❌ 目录不存在：{root_dir}")
        return
    
    # 查找所有 DayXX 文件夹
    day_folders = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("Day")])
    
    if not day_folders:
        print("❌ 没有找到 DayXX 文件夹")
        return
    
    print(f"✅ 找到 {len(day_folders)} 个 Day 文件夹\n")
    
    modified_count = 0
    
    for day_folder in day_folders:
        print(f"处理 {day_folder.name}...")
        
        # 查找该文件夹中的所有 md 文件
        md_files = sorted(day_folder.glob("*.md"))
        
        for md_file in md_files:
            # 跳过索引文件（可选）
            if "索引" in md_file.name or "index" in md_file.name.lower():
                print(f"  ⏭️  跳过索引文件：{md_file.name}")
                continue
            
            # 读取文件内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否已经包含引流文案
            if "关于作者 & 获取更多资源" in content:
                print(f"  ✅ 已包含引流文案：{md_file.name}")
                continue
            
            # 添加引流文案
            with open(md_file, 'a', encoding='utf-8') as f:
                f.write(ADDITIONAL_CONTENT)
            
            modified_count += 1
            print(f"  ✏️  已添加：{md_file.name}")
    
    print(f"\n{'='*50}")
    print(f"✅ 批量处理完成！共修改 {modified_count} 个文件")
    print(f"{'='*50}")


if __name__ == "__main__":
    # 项目根目录
    PROJECT_ROOT = r"e:\learn\AI 入门 30 天挑战"
    
    # DayXX 文件夹所在目录
    DAY_FOLDER_DIR = os.path.join(PROJECT_ROOT, "tutorials")
    
    print("开始批量添加公众号引流文案...")
    print("=" * 50)
    
    add_ad_to_markdown_files(DAY_FOLDER_DIR)
    
    print("\n💡 提示：")
    print("1. 已跳过包含'索引'字样的文件")
    print("2. 如果某文件已包含引流文案，会自动跳过")
    print("3. 建议先在 1-2 个文件上测试，确认效果后再批量处理")
