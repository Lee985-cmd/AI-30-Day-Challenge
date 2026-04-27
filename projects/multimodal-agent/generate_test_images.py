"""
生成测试图片脚本
创建用于多模态Agent测试的示例图片
"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_product_image(filename, product_name, color="white", text_color="black"):
    """
    创建产品示意图
    
    Args:
        filename: 文件名
        product_name: 产品名称
        color: 背景色
        text_color: 文字颜色
    """
    # 创建图像
    img = Image.new('RGB', (800, 600), color=color)
    draw = ImageDraw.Draw(img)
    
    # 绘制产品框
    draw.rectangle([100, 100, 700, 500], outline=text_color, width=3)
    
    # 添加产品名称
    try:
        font = ImageFont.truetype("simhei.ttf", 40)  # Windows黑体
    except:
        font = ImageFont.load_default()
    
    # 计算文字位置（居中）
    text_bbox = draw.textbbox((0, 0), product_name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (800 - text_width) // 2
    y = (600 - text_height) // 2
    
    draw.text((x, y), product_name, fill=text_color, font=font)
    
    # 保存
    filepath = os.path.join("test_images", filename)
    img.save(filepath)
    print(f"✅ 已创建: {filepath}")
    
    return filepath


def create_problem_image(filename, problem_text, bg_color="lightyellow"):
    """
    创建问题示意图
    
    Args:
        filename: 文件名
        problem_text: 问题描述
        bg_color: 背景色
    """
    # 创建图像
    img = Image.new('RGB', (800, 600), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # 绘制警告标志
    draw.ellipse([350, 50, 450, 150], outline="red", width=5)
    draw.text((385, 75), "!", fill="red", font=None)
    
    # 添加问题描述
    try:
        font = ImageFont.truetype("simhei.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # 分行显示
    lines = []
    words = problem_text.split()
    current_line = ""
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] < 700:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # 绘制文字
    y_start = 200
    line_height = 40
    for i, line in enumerate(lines):
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        x = (800 - text_width) // 2
        y = y_start + i * line_height
        draw.text((x, y), line, fill="darkred", font=font)
    
    # 保存
    filepath = os.path.join("test_images", filename)
    img.save(filepath)
    print(f"✅ 已创建: {filepath}")
    
    return filepath


def main():
    """主函数"""
    print("=" * 60)
    print("生成多模态 Agent 测试图片")
    print("=" * 60)
    print()
    
    # 确保目录存在
    os.makedirs("test_images", exist_ok=True)
    
    print("📦 创建产品图片...")
    print("-" * 60)
    
    # 产品图片
    create_product_image("laptop.jpg", "笔记本电脑", color="lightblue")
    create_product_image("phone.jpg", "智能手机", color="lightgreen")
    create_product_image("headphones.jpg", "无线耳机", color="lightyellow")
    create_product_image("camera.jpg", "数码相机", color="lightcoral")
    create_product_image("watch.jpg", "智能手表", color="lightgray")
    
    print()
    print("⚠️  创建问题图片...")
    print("-" * 60)
    
    # 问题图片
    create_problem_image(
        "screen_crack.jpg",
        "屏幕出现裂纹 无法触摸",
        bg_color="mistyrose"
    )
    
    create_problem_image(
        "battery_issue.jpg",
        "电池续航短 充电慢",
        bg_color="lemonchiffon"
    )
    
    create_problem_image(
        "wifi_problem.jpg",
        "WiFi连接不稳定 经常断线",
        bg_color="lightcyan"
    )
    
    create_problem_image(
        "overheating.jpg",
        "设备过热 自动关机",
        bg_color="lavenderblush"
    )
    
    print()
    print("=" * 60)
    print("✅ 测试图片生成完成！")
    print("=" * 60)
    print()
    print("📁 图片位置: test_images/")
    print()
    print("💡 使用建议:")
    print("1. 产品识别模式: 上传 laptop.jpg, phone.jpg 等")
    print("2. 问题诊断模式: 上传 screen_crack.jpg, battery_issue.jpg 等")
    print("3. 智能对话模式: 任意图片 + 问题")
    print()
    print("🎯 测试步骤:")
    print("1. 启动 Streamlit: streamlit run streamlit_app.py")
    print("2. 选择模式")
    print("3. 上传图片")
    print("4. 输入问题或点击诊断")
    print("=" * 60)


if __name__ == "__main__":
    main()
