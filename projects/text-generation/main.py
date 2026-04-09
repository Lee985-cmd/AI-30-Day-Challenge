"""
文本生成项目 - 主程序

运行方法:
    # 训练模型
    python main.py --mode train --epochs 50
    
    # 生成文本
    python main.py --mode generate --prompt "床前明月光"
"""

import argparse
from train import train
from generate import generate


def main():
    parser = argparse.ArgumentParser(description='文本生成项目')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'generate'],
                       help='模式: train 或 generate')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--prompt', type=str, default='床前',
                       help='生成文本的开头')
    parser.add_argument('--length', type=int, default=50,
                       help='生成文本的长度')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='温度参数（控制创造性）')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args.epochs)
    elif args.mode == 'generate':
        generate(args.prompt, args.length, args.temperature)


if __name__ == '__main__':
    main()
