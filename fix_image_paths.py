"""
批量修复教程中的二维码图片路径
将 ../../../images/logos/ewm.jpg 改为 ../../images/logos/ewm.jpg
"""
import os
from pathlib import Path


def fix_image_paths(root_dir):
    """批量修复图片路径"""
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
    
    fixed_count = 0
    
    for day_folder in day_folders:
        print(f"处理 {day_folder.name}...")
        
        # 查找该文件夹中的所有 md 文件
        md_files = sorted(day_folder.glob("*.md"))
        
        for md_file in md_files:
            # 读取文件内容
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否需要修复
            if "../../../images/logos/ewm.jpg" in content:
                # 修复路径
                new_content = content.replace(
                    "../../../images/logos/ewm.jpg",
                    "../../images/logos/ewm.jpg"
                )
                
                # 写回文件
                with open(md_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                fixed_count += 1
                print(f"  ✅ 已修复：{md_file.name}")
            else:
                print(f"  ⏭️  无需修复：{md_file.name}")
    
    print(f"\n{'='*50}")
    print(f"✅ 批量修复完成！共修复 {fixed_count} 个文件")
    print(f"{'='*50}")


if __name__ == "__main__":
    # tutorials 目录
    TUTORIALS_DIR = r"e:\learn\AI 入门 30 天挑战\tutorials"
    
    print("开始批量修复二维码图片路径...")
    print("=" * 50)
    
    fix_image_paths(TUTORIALS_DIR)
    
    print("\n💡 提示：")
    print("1. 已将 ../../../images/logos/ewm.jpg 改为 ../../images/logos/ewm.jpg")
    print("2. 如果某文件不需要修复，会自动跳过")
