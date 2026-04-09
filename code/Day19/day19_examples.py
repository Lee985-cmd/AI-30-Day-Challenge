"""
Day19 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day19_examples.py

注意: 某些代码可能需要安装额外的库
"""

# 导入必要的库
import sys
import os

# 尝试导入常用库
try:
    import numpy as np
except ImportError:
    print("提示: 需要安装 numpy: pip install numpy")
    np = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("提示: 需要安装 matplotlib: pip install matplotlib")
    plt = None

try:
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("提示: 需要安装 scikit-learn: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("提示: 需要安装 PyTorch: pip install torch torchvision")

print("=" * 60)
print("Day19 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

# ============================================================================
# 第一部分：导入必要的库
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置随机种子，让结果可以复现
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("DCGAN 生成手写数字 - 从零开始")
print("=" * 60)

# ============================================================================
# 第二部分：准备数据
# ============================================================================

"""
MNIST 数据集是什么？
- 60000 张手写数字图片 (0-9)
- 每张图片 28x28 像素 (黑白)
- AI 界的"Hello World",新手必学

就像小学生用的数字字帖，让 AI 学习认数字
"""

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 放大到 64x64 (方便卷积操作)
    transforms.ToTensor(),         # 转成 Tensor
    transforms.Normalize([0.5], [0.5])  # 归一化到 [-1, 1]
])

# 下载并加载数据
print("\n正在下载 MNIST 数据集...")
print("提示：第一次运行会自动下载，大概 10MB")

train_dataset = datasets.MNIST(
    root='./data/mnist',      # 数据存放位置
    train=True,                # 用训练集
    download=True,            # 自动下载
    transform=transform       # 预处理
)

# 创建数据加载器
batch_size = 128  # 一次看 128 张图片

dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,  # 打乱顺序
    num_workers=0  # Windows 用户设为 0
)

print(f"✓ 数据加载完成!")
print(f"  - 总图片数：{len(train_dataset)}")
print(f"  - 批次大小：{batch_size}")
print(f"  - 批次数：{len(dataloader)}")

# ===== 代码块 2 =====

# ============================================================================
# 第三部分：创建生成器 G - "造假币的人"
# ============================================================================

"""
生成器的任务:
输入：一堆随机噪声 (看不出规律)
输出：一张像模像样的手写数字图片

就像魔术师：
输入：一把沙子 (随机噪声)
输出：一只鸽子 (清晰的图片)

架构设计思路 (从下往上):
1. 先从一个点开始 (1x1)
2. 逐渐放大 (4x4 → 8x8 → 16x16 → 32x32 → 64x64)
3. 最后变成完整的图片

用什么层？
- ConvTranspose2d：转置卷积，专门用来放大的
- BatchNorm：批归一化，让训练更稳定
- ReLU：激活函数，增加非线性
- Tanh：最后一层，把值限制在 [-1, 1]
"""

class Generator(nn.Module):
    """生成器网络"""
    
    def __init__(self):
        super(Generator, self).__init__()
        
        # 输入：随机噪声向量 (100 维)
        # 输出：64x64 的单通道图片
        
        # 第一步：从 100 维噪声到一个小方块 (4x4)
        self.layer1 = nn.Sequential(
            # 输入：(batch_size, 100, 1, 1)
            nn.ConvTranspose2d(
                in_channels=100,   # 输入通道数 (噪声维度)
                out_channels=512,  # 输出通道数 (特征图数量)
                kernel_size=4,     # 卷积核大小 4x4
                stride=1,          # 步长
                padding=0          # 填充
            ),
            # 输出：(batch_size, 512, 4, 4)
            
            nn.BatchNorm2d(512),   # 批归一化
            nn.ReLU(True)          # 激活函数
        )
        
        # 第二步：从 4x4 放大到 8x8
        self.layer2 = nn.Sequential(
            # 输入：(batch_size, 512, 4, 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            # kernel_size=4, stride=2, padding=1
            # 输出尺寸计算公式：(输入 -1)*stride - 2*padding + kernel_size
            # = (4-1)*2 - 2*1 + 4 = 8
            
            nn.BatchNorm2d(256),
            nn.ReLU(True)
            # 输出：(batch_size, 256, 8, 8)
        )
        
        # 第三步：从 8x8 放大到 16x16
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
            # 输出：(batch_size, 128, 16, 16)
        )
        
        # 第四步：从 16x16 放大到 32x32
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
            # 输出：(batch_size, 64, 32, 32)
        )
        
        # 第五步：从 32x32 放大到 64x64，并变成单通道图片
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            # 注意：这里输出通道是 1 (灰度图)
            
            nn.Tanh()
            # Tanh 把值限制在 [-1, 1],和我们的数据预处理对应
            # 输出：(batch_size, 1, 64, 64)
        )
        
    def forward(self, x):
        """前向传播"""
        # x 的形状：(batch_size, 100, 1, 1)
        
        x = self.layer1(x)  # → (batch_size, 512, 4, 4)
        x = self.layer2(x)  # → (batch_size, 256, 8, 8)
        x = self.layer3(x)  # → (batch_size, 128, 16, 16)
        x = self.layer4(x)  # → (batch_size, 64, 32, 32)
        x = self.layer5(x)  # → (batch_size, 1, 64, 64)
        
        return x

# 测试生成器
G = Generator()
print("\n✓ 生成器创建成功!")

# 创建一个测试输入
test_noise = torch.randn(1, 100, 1, 1)  # 随机噪声
test_output = G(test_noise)
print(f"  - 输入形状：{test_noise.shape}")
print(f"  - 输出形状：{test_output.shape}")
print(f"  - 输出范围：[{test_output.min():.3f}, {test_output.max():.3f}]")

# ===== 代码块 3 =====

# ============================================================================
# 第四部分：创建判别器 D - "警察"
# ============================================================================

"""
判别器的任务:
输入：一张图片 (可能是真的，也可能是 G 生成的假图片)
输出：一个概率值 (0~1)
  - 接近 1：认为是真的
  - 接近 0：认为是假的

架构设计思路 (从上往下):
1. 从大图片开始 (64x64)
2. 逐渐缩小 (32x32 → 16x16 → 8x8 → 4x4)
3. 最后变成一个数字 (概率)

用什么层？
- Conv2d：普通卷积，用来提取特征
- LeakyReLU：泄漏 ReLU，比 ReLU 更适合判别器
- Sigmoid：最后一层，输出概率
"""

class Discriminator(nn.Module):
    """判别器网络"""
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        # 输入：64x64 的单通道图片
        # 输出：一个概率值 (标量)
        
        # 第一步：从 64x64 缩小到 32x32
        self.layer1 = nn.Sequential(
            # 输入：(batch_size, 1, 64, 64)
            nn.Conv2d(
                in_channels=1,     # 输入通道 (灰度图)
                out_channels=64,   # 输出通道
                kernel_size=4,     # 卷积核 4x4
                stride=2,          # 步长 2 (缩小一半)
                padding=1          # 填充
            ),
            # 输出尺寸：(64-4)/2 + 1 = 32
            # 输出：(batch_size, 64, 32, 32)
            
            nn.LeakyReLU(0.2, inplace=True)
            # LeakyReLU: 允许负的梯度，防止"神经元死亡"
            # 0.2 表示负值区域也有 20% 的梯度
        )
        
        # 第二步：从 32x32 缩小到 16x16
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
            # 输出：(batch_size, 128, 16, 16)
        )
        
        # 第三步：从 16x16 缩小到 8x8
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
            # 输出：(batch_size, 256, 8, 8)
        )
        
        # 第四步：从 8x8 缩小到 4x4
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
            # 输出：(batch_size, 512, 4, 4)
        )
        
        # 第五步：从 4x4 变成一个数字
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1, 4, 1, 0),
            # kernel_size=4, stride=1, padding=0
            # 输出：(batch_size, 1, 1, 1)
            
            nn.Sigmoid()
            # Sigmoid 输出 0~1 之间的值，表示"真实"的概率
        )
        
    def forward(self, x):
        """前向传播"""
        # x 的形状：(batch_size, 1, 64, 64)
        
        x = self.layer1(x)  # → (batch_size, 64, 32, 32)
        x = self.layer2(x)  # → (batch_size, 128, 16, 16)
        x = self.layer3(x)  # → (batch_size, 256, 8, 8)
        x = self.layer4(x)  # → (batch_size, 512, 4, 4)
        x = self.layer5(x)  # → (batch_size, 1, 1, 1)
        
        # 展平成一个数字
        x = x.view(x.size(0), -1)  # → (batch_size, 1)
        
        return x

# 测试判别器
D = Discriminator()
print("\n✓ 判别器创建成功!")

# 创建一个测试输入
test_img = torch.randn(1, 1, 64, 64)  # 随机图片
test_output = D(test_img)
print(f"  - 输入形状：{test_img.shape}")
print(f"  - 输出形状：{test_output.shape}")
print(f"  - 输出值：{test_output.item():.3f} (接近 0 表示假，接近 1 表示真)")

# ===== 代码块 4 =====

# ============================================================================
# 第五部分：定义损失函数和优化器
# ============================================================================

"""
损失函数：二元交叉熵 BCELoss

干什么用？
- 衡量"预测值"和"真实值"的差距
- 值越小，说明预测越准

公式 (不用懂，看看就好):
BCE = -[y*log(p) + (1-y)*log(1-p)]

说人话:
- 如果是真图片 (y=1)，希望 D 输出 p 接近 1
- 如果是假图片 (y=0)，希望 D 输出 p 接近 0
"""

# 二元交叉熵损失
criterion = nn.BCELoss()

# 优化器
# G 和 D 分别优化自己的参数
optimizer_G = optim.Adam(
    G.parameters(), 
    lr=0.0002,        # 学习率 (别太大，会炸)
    betas=(0.5, 0.999)  # Adam 的参数
)

optimizer_D = optim.Adam(
    D.parameters(), 
    lr=0.0002,
    betas=(0.5, 0.999)
)

print("\n✓ 损失函数和优化器设置完成!")
print(f"  - 损失函数：BCELoss (二元交叉熵)")
print(f"  - 优化器：Adam (lr=0.0002, betas=(0.5, 0.999))")

# ===== 代码块 5 =====

# ============================================================================
# 第六部分：开始训练！重头戏来了!
# ============================================================================

"""
训练流程 (记住这个节奏):

对于每一个 epoch (轮):
    对于每一批数据 (batch):
        
        【第 1 步：训练判别器 D】
        1. 用真实图片训练 D
           - 告诉 D：这是真的！
           - D 应该输出接近 1
        
        2. 用假图片训练 D
           - 让 G 生成假图片
           - 告诉 D：这是假的!
           - D 应该输出接近 0
        
        3. 更新 D 的参数
        
        【第 2 步：训练生成器 G】
        4. 让 G 生成假图片
        5. 骗过 D，让 D 以为是真的
           - D 应该输出接近 1
        
        6. 更新 G 的参数

关键技巧:
✓ G 和 D 交替训练，不要一起更新
✓ D 训练 1-2 次，G 训练 1 次
✓ 标签平滑：真实标签用 0.9 而不是 1.0 (防止过拟合)
✓ 噪声采样：每次重新生成随机噪声
"""

# 训练参数
num_epochs = 50  # 训练 50 轮 (可以调整)
sample_dir = 'gan_samples'  # 保存生成图片的文件夹

print("\n" + "=" * 60)
print("开始训练 GAN!")
print("=" * 60)
print(f"训练参数:")
print(f"  - 轮数：{num_epochs}")
print(f"  - 批次大小：{batch_size}")
print(f"  - 学习率：0.0002")
print("=" * 60)

# 记录损失值，方便画图
G_losses = []
D_losses = []

# 固定一批噪声，用来观察生成效果的变化
fixed_noise = torch.randn(16, 100, 1, 1)

for epoch in range(num_epochs):
    print(f"\n【Epoch {epoch+1}/{num_epochs}】")
    
    # 初始化本 epoch 的损失
    epoch_G_loss = 0
    epoch_D_loss = 0
    num_batches = 0
    
    for batch_idx, (real_images, _) in enumerate(dataloader):
        batch_size_now = real_images.size(0)
        
        # ===== 【第 1 步：训练判别器 D】=====
        
        D.zero_grad()  # 清空梯度
        
        # --- 用真实图片训练 ---
        real_labels = torch.ones(batch_size_now, 1)  # 全 1 向量 (真的)
        real_labels = real_labels * 0.9  # 标签平滑
        
        output_real = D(real_images)  # D 对真实图片的判断
        D_loss_real = criterion(output_real, real_labels)
        
        # --- 用假图片训练 ---
        noise = torch.randn(batch_size_now, 100, 1, 1)  # 随机噪声
        fake_images = G(noise)  # G 生成假图片
        
        fake_labels = torch.zeros(batch_size_now, 1)  # 全 0 向量 (假的)
        output_fake = D(fake_images.detach())  # D 对假图片的判断
        # .detach() 很重要！阻止梯度传到 G
        
        D_loss_fake = criterion(output_fake, fake_labels)
        
        # --- 合并损失 ---
        D_loss = D_loss_real + D_loss_fake
        
        # --- 反向传播 ---
        D_loss.backward()
        optimizer_D.step()  # 更新 D 的参数
        
        # ===== 【第 2 步：训练生成器 G】=====
        
        G.zero_grad()  # 清空梯度
        
        # 目标：骗过 D，让 D 以为假图片是真的
        labels_for_G = torch.ones(batch_size_now, 1)  # 目标是让 D 输出 1
        
        noise = torch.randn(batch_size_now, 100, 1, 1)
        fake_images = G(noise)
        
        output_fake_for_G = D(fake_images)  # D 的判断
        G_loss = criterion(output_fake_for_G, labels_for_G)
        
        # --- 反向传播 ---
        G_loss.backward()
        optimizer_G.step()  # 更新 G 的参数
        
        # 记录损失
        epoch_G_loss += G_loss.item()
        epoch_D_loss += D_loss.item()
        num_batches += 1
        
        # 每 100 个 batch 打印一次进度
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)} | "
                  f"D 损失：{D_loss.item():.4f} | "
                  f"G 损失：{G_loss.item():.4f}")
    
    # 【Epoch 结束：统计和可视化】
    
    avg_G_loss = epoch_G_loss / num_batches
    avg_D_loss = epoch_D_loss / num_batches
    
    G_losses.append(avg_G_loss)
    D_losses.append(avg_D_loss)
    
    print(f"\n【Epoch {epoch+1} 完成】")
    print(f"  平均 G 损失：{avg_G_loss:.4f}")
    print(f"  平均 D 损失：{avg_D_loss:.4f}")
    
    # 生成并保存图片
    with torch.no_grad():  # 不需要计算梯度
        fake_samples = G(fixed_noise)
        fake_samples = (fake_samples + 1) / 2  # 从 [-1, 1] 转回 [0, 1]
        
        # 画成网格
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                img = fake_samples[idx].squeeze()
                axes[i][j].imshow(img, cmap='gray')
                axes[i][j].axis('off')
        
        plt.suptitle(f'Epoch {epoch+1} - G 生成的图片', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{sample_dir}/epoch_{epoch+1:03d}.png', dpi=150)
        plt.close()
    
    if (epoch + 1) % 5 == 0:
        print(f"  ✓ 图片已保存")

# ===== 代码块 6 =====

"""
关键参数:

1. 学习率 (learning rate)
   - 推荐：0.0002
   - 太大：训练不稳定
   - 太小：收敛慢

2. 批次大小 (batch size)
   - 推荐：64-128
   - 太大：内存不够
   - 太小：训练不稳定

3. 噪声维度 (noise dimension)
   - 推荐：100-200
   - 太小：生成单一
   - 太大：难以训练

4. 训练轮数 (epochs)
   - MNIST: 50-100
   - CIFAR-10: 100-200
   - 高分辨率：200+

# ===== 代码块 7 =====

# 试试把噪声从 100 改成 50 或 200
# 观察生成效果有什么变化
noise_dim = 50  # 或 200

# ===== 代码块 8 =====

# 在生成器中加一层或减一层
# 看看对生成质量的影响

# ===== 代码块 9 =====

# 生成彩色图片
train_dataset = datasets.CIFAR10(
    root='./data/cifar10',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
)

# 修改网络以适应 3 通道图片
# Generator 输出改为 3 通道
# Discriminator 输入改为 3 通道

# ===== 代码块 10 =====

# 把 ReLU 换成 LeakyReLU 或 ELU
# 比较效果差异

# ===== 代码块 11 =====

# 训练 D 时:
output_fake = D(fake_images.detach())

原因:
- 我们只想更新 D 的参数，不想影响 G
- .detach() 阻止梯度传回 G

就像考试:
- 老师打分时，只评价学生的答案
- 不应该反过来影响学生的学习方法
- 学生学习方法是另一回事 (G 的训练)