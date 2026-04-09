"""
目标检测项目 - 使用 YOLOv8

运行方法:
    # 检测图片
    python detect.py --source image.jpg
    
    # 实时摄像头检测
    python detect.py --source 0
    
    # 检测视频
    python detect.py --source video.mp4
"""

import argparse
from ultralytics import YOLO
import cv2
import os


def detect_image(model, source, conf=0.25):
    """检测图片"""
    print(f"\n🔍 检测图片: {source}")
    
    # 运行检测
    results = model(source, conf=conf)
    
    # 显示结果
    for result in results:
        # 打印检测结果
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"✅ 检测到 {len(boxes)} 个物体:")
            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf_score = float(box.conf[0])
                name = result.names[cls]
                print(f"   {i+1}. {name} (置信度: {conf_score:.2f})")
        else:
            print("ℹ️  未检测到物体")
        
        # 保存带标注的图片
        annotated = result.plot()
        output_path = f"result_{os.path.basename(source)}"
        cv2.imwrite(output_path, annotated)
        print(f"✅ 结果已保存: {output_path}")
        
        # 显示图片（如果支持）
        try:
            cv2.imshow('Detection Result', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            pass


def detect_video(model, source, conf=0.25):
    """检测视频或摄像头"""
    print(f"\n🎥 开始检测: {source}")
    
    # 打开视频或摄像头
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ 无法打开: {source}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps} FPS")
    
    # 创建输出视频
    if source != '0':  # 不是摄像头
        output_path = f"result_{os.path.basename(source)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        results = model(frame, conf=conf, verbose=False)
        
        # 绘制结果
        annotated = results[0].plot()
        
        # 显示帧率
        frame_count += 1
        cv2.putText(annotated, f'FPS: {frame_count/max(1, frame_count)*30:.1f}',
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示
        cv2.imshow('Real-time Detection', annotated)
        
        # 保存视频
        if source != '0':
            out.write(annotated)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    if source != '0':
        out.release()
    cv2.destroyAllWindows()
    
    if source != '0':
        print(f"✅ 结果已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv8 目标检测')
    parser.add_argument('--source', type=str, default='image.jpg',
                       help='输入源：图片路径、视频路径、或 0（摄像头）')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='模型名称: yolov8n/s/m/l/x')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 (0-1)')
    parser.add_argument('--save', action='store_true',
                       help='保存结果')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("YOLOv8 目标检测")
    print("=" * 60)
    print(f"模型: {args.model}")
    print(f"输入: {args.source}")
    print(f"置信度阈值: {args.conf}")
    print("=" * 60)
    
    # 加载模型
    print("\n📦 加载模型...")
    model = YOLO(args.model)
    print("✅ 模型加载完成")
    
    # 判断输入类型
    if args.source.isdigit():
        # 摄像头
        detect_video(model, int(args.source), args.conf)
    elif args.source.endswith(('.mp4', '.avi', '.mov')):
        # 视频
        detect_video(model, args.source, args.conf)
    else:
        # 图片
        if os.path.exists(args.source):
            detect_image(model, args.source, args.conf)
        else:
            print(f"❌ 文件不存在: {args.source}")
            print("\n💡 提示: 可以使用以下命令测试:")
            print("   python detect.py --source 0  # 摄像头")
            print("   python detect.py --source test.jpg  # 图片")


if __name__ == '__main__':
    main()
