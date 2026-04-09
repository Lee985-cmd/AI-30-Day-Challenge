"""
目标检测训练脚本

运行方法:
    python train.py
"""

from ultralytics import YOLO


def main():
    print("=" * 60)
    print("YOLOv8 自定义训练")
    print("=" * 60)
    
    # 加载预训练模型
    print("\n📦 加载预训练模型...")
    model = YOLO('yolov8n.pt')
    print("✅ 模型加载完成")
    
    # 训练参数
    print("\n🚀 开始训练...")
    results = model.train(
        data='dataset.yaml',  # 数据集配置
        epochs=100,            # 训练轮数
        imgsz=640,             # 图片尺寸
        batch=16,              # 批次大小
        device=0 if torch.cuda.is_available() else 'cpu',  # 设备
        workers=4,             # 数据加载线程数
        patience=20,           # 早停耐心值
        save=True,             # 保存检查点
        plots=True,            # 绘制训练曲线
    )
    
    print("\n✅ 训练完成！")
    print(f"最佳模型保存在: {results.save_dir}")
    
    # 验证模型
    print("\n📊 验证模型...")
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")


if __name__ == '__main__':
    import torch
    main()
