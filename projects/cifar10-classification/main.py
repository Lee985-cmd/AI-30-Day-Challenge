"""
CIFAR-10 图像分类项目 - 主程序入口

运行方法:
    python main.py

功能:
    - 训练模型
    - 评估模型
    - 可视化结果
"""

import argparse
from train import train
from evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 图像分类')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'],
                       help='模式: train 或 evaluate')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate_model()


if __name__ == '__main__':
    main()
