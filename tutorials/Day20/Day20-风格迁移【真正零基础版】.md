# 🎨 Day20: 风格迁移 - 把照片变成名画【真正零基础版】

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **让 AI 成为艺术家！把你的照片变成梵高、毕加索风格!**  
> **本教程：完整可运行代码 + 超详细讲解 + 实战项目**

---

## 📚 目录

1. [风格迁移是什么？](#风格迁移是什么)
2. [核心原理 - Gram 矩阵的魔力](#核心原理 -gram 矩阵的魔力)
3. [实战：照片变梵高风格](#实战：照片变梵高风格)
4. [快速风格迁移](#快速风格迁移)
5. [完整项目：制作你的艺术滤镜](#完整项目：制作你的艺术滤镜)
6. [常见问题](#常见问题)

---

## 🤔 风格迁移是什么？

### 说人话版本

想象一下这个场景:

```
你有一张:
- 你家的宠物狗照片 (内容)

你想把它变成:
- 梵高的星空风格 (风格)

结果:
- 还是你的狗 (内容不变)
- 但画风是梵高的笔触 (风格变了)

就像:
- 同样的风景，用不同的相机滤镜
- 同样的人，用不同的绘画风格
```

**这就是风格迁移!**

- **内容图片** = 你要转换的图片 (比如你的照片)
- **风格图片** = 你想模仿的艺术风格 (比如梵高的画)
- **生成图片** = 结果 (你的照片 + 梵高风格)

### 风格迁移能做什么？

**真实应用场景:**

1. **艺术创作**
   - 把照片变世界名画
   - 制作个性化艺术品
   - 创意摄影后期

2. **影视后期**
   - 电影特效
   - 动画片制作
   - 游戏美术

3. **手机 APP**
   - Prisma(最火的风格迁移 APP)
   - 美图秀秀艺术滤镜
   - 抖音特效

4. **设计行业**
   - 平面设计
   - 室内设计
   - 服装设计

---

## 🔬 核心原理 - Gram 矩阵的魔力

### 神经网络看到了什么？

```
一张图片输入到 CNN(卷积神经网络):

第 1 层：看到边缘、线条
         ↓
第 2 层：看到纹理、图案
         ↓
第 3 层：看到物体部分 (眼睛、轮子)
         ↓
第 4 层：看到完整物体 (人脸、汽车)

越深的层，看到的越抽象
```

### 内容和风格怎么分离？

```python
"""
内容的表示:
- 用深层的特征 (能看到物体整体)
- 保留图片的"内容信息"

风格的表示:
- 用浅层的特征 (纹理、颜色)
- 用 Gram 矩阵计算"风格信息"

Gram 矩阵是什么？
说人话：
- 计算不同特征之间的相关性
- 比如：某些纹理总是同时出现
- 这些"同时出现"的模式就是风格
"""
```

### Gram 矩阵计算 (不用懂数学，看看就好)

```python
import torch

def gram_matrix(x):
    """
    计算 Gram 矩阵
    
    输入：特征图 (batch_size, channels, height, width)
    输出：Gram 矩阵 (channels, channels)
    
    说人话:
    - 看每个通道 (特征) 之间有多"像"
    - 像的就是风格
    """
    batch_size, channels, height, width = x.shape
    
    # 把特征图拉直
    features = x.view(batch_size * channels, height * width)
    
    # 计算相关性矩阵
    G = torch.mm(features, features.t())
    
    # 归一化
    return G.div(batch_size * channels * height * width)

"""
例子:

假设有两个特征:
- 特征 A: 检测竖线
- 特征 B: 检测横线

如果 A 和 B 总是一起出现
→ Gram 矩阵中 A-B 的值很大
→ 说明这种"横竖交叉"是风格特点

梵高的画:
- 短促的笔触
- 旋转的纹理
- 强烈的色彩对比

这些都体现在 Gram 矩阵中!
"""
```

---

## 🎨 实战：照片变梵高风格

### 方法一：优化输入图片 (经典方法)

```python
# ============================================================================
# 第一部分：导入库
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

print("=" * 60)
print("神经风格迁移 - 把照片变成名画")
print("=" * 60)

# ============================================================================
# 第二部分：加载预训练模型
# ============================================================================

"""
用什么模型？
- VGG19: 效果最好，最常用
- ResNet: 也可以，但 VGG 更适合风格迁移

为什么用预训练模型？
- 在 ImageNet 上训练过，认识各种物体
- 特征提取能力强
- 不用自己训练，省时省力
"""

# 加载 VGG19 模型
cnn = models.vgg19(pretrained=True).features

# 冻结参数 (不需要更新)
for param in cnn.parameters():
    param.requires_grad_(False)

print(f"✓ VGG19 加载成功!")
print(f"  - 网络层数：{len(cnn)}")

# ============================================================================
# 第三部分：准备图片
# ============================================================================

"""
需要三张图片:
1. 内容图片 - 你的照片
2. 风格图片 - 你想模仿的名画
3. 生成图片 - 从噪声开始，逐渐优化
"""

# 图片预处理
def load_image(image_path, size=512):
    """加载并预处理图片"""
    
    image = Image.open(image_path)
    
    # 转成 RGB(如果是 RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((size, size)),  # 调整大小
        transforms.ToTensor(),             # 转 Tensor
        transforms.Normalize(              # 归一化
            mean=[0.485, 0.456, 0.406],   # ImageNet 的平均值
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 添加 batch 维度 (1, 3, H, W)
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

# 下载示例图片 (如果没有，用示例链接)
print("\n准备图片...")

# 你可以替换成自己的图片路径
content_path = 'my_photo.jpg'      # 你的照片
style_path = 'starry_night.jpg'    # 梵高的星空

# 检查文件是否存在，不存在就创建示例
if not os.path.exists(content_path):
    print(f"提示：请把您的照片放到 '{content_path}'")
    print("或者使用下面的在线示例")
    
    # 创建一个示例图片 (纯色)
    content_img = Image.new('RGB', (512, 512), color='skyblue')
    content_img.save(content_path)

if not os.path.exists(style_path):
    print(f"提示：请下载梵高的星空到 '{style_path}'")
    print("下载地址：https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg")
    
    # 创建一个示例风格图片 (随机纹理)
    style_img = Image.new('RGB', (512, 512))
    pixels = style_img.load()
    for i in range(512):
        for j in range(512):
            pixels[i, j] = (np.random.randint(0, 255), 
                           np.random.randint(0, 255), 
                           np.random.randint(0, 255))
    style_img.save(style_path)

# 加载图片
content_image = load_image(content_path, size=512)
style_image = load_image(style_path, size=512)

print(f"✓ 图片加载成功!")
print(f"  - 内容图片：{content_path} {content_image.shape}")
print(f"  - 风格图片：{style_path} {style_image.shape}")

# 显示原图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(content_image.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
axes[0].set_title('内容图片 (你的照片)', fontsize=14)
axes[0].axis('off')

axes[1].imshow(style_image.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
axes[1].set_title('风格图片 (梵高星空)', fontsize=14)
axes[1].axis('off')

plt.suptitle('准备开始风格迁移!', fontsize=16)
plt.tight_layout()
plt.savefig('input_images.png', dpi=150)
plt.show()

# ============================================================================
# 第四部分：定义损失函数
# ============================================================================

"""
三个关键损失:

1. 内容损失 (Content Loss)
   - 让生成图片和内容图片"长得像"
   - 比较深层的特征

2. 风格损失 (Style Loss)
   - 让生成图片和风格图片"画风像"
   - 比较 Gram 矩阵

3. 总损失 = 内容损失 + 风格损失 × 权重
   - 权重越大，风格越明显
   - 权重越小，内容越清晰
"""

class ContentLoss(nn.Module):
    """内容损失"""
    
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # 保存目标特征 (内容图片的特征)
        self.target = target.detach()  # .detach() 防止梯度传到 target
        
    def forward(self, x):
        # 计算当前特征和目标特征的差距
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    """风格损失"""
    
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # 计算目标 Gram 矩阵
        self.target = gram_matrix(target_feature).detach()
        
    def forward(self, x):
        # 计算当前 Gram 矩阵
        G = gram_matrix(x)
        
        # 比较 Gram 矩阵的差距
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

def gram_matrix(x):
    """计算 Gram 矩阵"""
    batch_size, channels, height, width = x.shape
    
    # 拉直特征图
    features = x.view(batch_size * channels, height * width)
    
    # 计算 Gram 矩阵
    G = torch.mm(features, features.t())
    
    # 归一化
    return G.div(batch_size * channels * height * width)

# ============================================================================
# 第五部分：选择特征层
# ============================================================================

"""
VGG19 有很多层，用哪些层来提取特征？

内容层:
- 选深层的层 (能看到物体整体)
- 推荐：conv_4_2 (第 23 层)

风格层:
- 选多层组合 (捕捉不同尺度的纹理)
- 推荐：conv_1_1, conv_2_1, conv_3_1, conv_4_1, conv_5_1
"""

# VGG19 的层名称
# 0: conv1_1, 5: conv2_1, 10: conv3_1, 19: conv4_1, 28: conv5_1

content_layers_default = ['23']  # conv4_2
style_layers_default = ['0', '5', '10', '19', '28']  # conv1_1 到 conv5_1

print(f"\n选择的特征层:")
print(f"  - 内容层：{content_layers_default}")
print(f"  - 风格层：{style_layers_default}")

# ============================================================================
# 第六部分：构建风格迁移网络
# ============================================================================

"""
思路:
1. 把 VGG19 的层重新打包
2. 插入我们的损失函数
3. 输入图片，一层层跑，自动计算损失
"""

class StyleTransfer(nn.Module):
    """风格迁移网络"""
    
    def __init__(self, cnn, content_layers, style_layers, 
                 content_weight=1.0, style_weight=1000000.0):
        super(StyleTransfer, self).__init__()
        
        self.cnn = cnn
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        # 存储损失函数和内容/风格目标
        self.content_losses = []
        self.style_losses = []
        
        # 遍历 VGG 的每一层
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                name = f"conv_{i}"
                self.add_module(name, layer)
                
                # 如果是内容层
                if str(i) in content_layers:
                    target = None  # 后面会设置
                    content_loss = ContentLoss(target)
                    self.add_module(f"content_loss_{i}", content_loss)
                    self.content_losses.append(content_loss)
                
                # 如果是风格层
                if str(i) in style_layers:
                    target = None  # 后面会设置
                    style_loss = StyleLoss(target)
                    self.add_module(f"style_loss_{i}", style_loss)
                    self.style_losses.append(style_loss)
                
                i += 1
            
            elif isinstance(layer, nn.ReLU):
                name = f"relu_{i}"
                self.add_module(name, layer)
                i += 1
            
            elif isinstance(layer, nn.MaxPool2d):
                name = f"pool_{i}"
                self.add_module(name, layer)
                i += 1
            
            elif isinstance(layer, nn.BatchNorm2d):
                name = f"bn_{i}"
                self.add_module(name, layer)
                i += 1
    
    def forward(self, x):
        """前向传播"""
        self.content_losses = []
        self.style_losses = []
        
        i = 0
        for name, module in self.named_children():
            if isinstance(module, (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.BatchNorm2d)):
                x = module(x)
                
                # 遇到内容损失层
                if name.startswith('content_loss'):
                    layer_num = int(name.split('_')[-1])
                    if str(layer_num) in self.content_layers:
                        target = getattr(self, f'content_loss_{layer_num}').target
                        content_loss = ContentLoss(target)
                        setattr(self, f'content_loss_{layer_num}', content_loss)
                        self.content_losses.append(content_loss)
                        x = content_loss(x)
                
                # 遇到风格损失层
                if name.startswith('style_loss'):
                    layer_num = int(name.split('_')[-1])
                    if str(layer_num) in self.style_layers:
                        target = getattr(self, f'style_loss_{layer_num}').target
                        style_loss = StyleLoss(target)
                        setattr(self, f'style_loss_{layer_num}', style_loss)
                        self.style_losses.append(style_loss)
                        x = style_loss(x)
        
        return x

# ============================================================================
# 第七部分：开始风格迁移!
# ============================================================================

"""
训练流程:

1. 从内容图片开始 (或者从噪声开始)
2. 输入到 VGG 网络
3. 计算每一层的特征
4. 计算内容损失和风格损失
5. 反向传播，更新输入图片
6. 重复 2-5 直到满意

注意:
- 不更新网络参数!
- 只更新输入图片!
"""

print("\n" + "=" * 60)
print("开始风格迁移!")
print("=" * 60)

# 超参数
num_steps = 300  # 优化步数
style_weight = 1e6  # 风格权重
content_weight = 1.0  # 内容权重
learning_rate = 0.01  # 学习率

print(f"训练参数:")
print(f"  - 优化步数：{num_steps}")
print(f"  - 风格权重：{style_weight}")
print(f"  - 内容权重：{content_weight}")
print(f"  - 学习率：{learning_rate}")

# 创建输入图片 (从内容图片或噪声开始)
input_image = content_image.clone().requires_grad_(True)  # 需要梯度

# 创建风格迁移模型
model = StyleTransfer(
    cnn=cnn,
    content_layers=content_layers_default,
    style_layers=style_layers_default,
    content_weight=content_weight,
    style_weight=style_weight
)

# 设置内容目标
for i, layer in enumerate(model.cnn):
    if str(i) in content_layers_default:
        with torch.no_grad():
            content_feature = model.cnn[:i+1](content_image)
            # 找到对应的损失层
            for loss_layer in model.modules():
                if isinstance(loss_layer, ContentLoss):
                    loss_layer.target = content_feature
                    break

# 设置风格目标
for i, layer in enumerate(model.cnn):
    if str(i) in style_layers_default:
        with torch.no_grad():
            style_feature = model.cnn[:i+1](style_image)
            # 找到对应的损失层
            for loss_layer in model.modules():
                if isinstance(loss_layer, StyleLoss):
                    loss_layer.target = style_feature
                    break

# 优化器 (只优化输入图片!)
optimizer = optim.LBFGS([input_image], lr=learning_rate)

# 记录损失
losses = []

print("\n开始优化...")

step = 0
while step <= num_steps:
    
    def closure():
        global step
        
        # 限制像素范围 [-1, 1]
        with torch.no_grad():
            input_image.clamp_(-1, 1)
        
        # 前向传播
        output = model(input_image)
        
        # 计算总损失
        content_score = 0
        style_score = 0
        
        for content_loss in model.content_losses:
            content_score += content_loss.loss
        
        for style_loss in model.style_losses:
            style_score += style_loss.loss
        
        total_loss = content_weight * content_score + style_weight * style_score
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 记录
        if step % 50 == 0:
            losses.append(total_loss.item())
            print(f"Step {step}: "
                  f"总损失={total_loss.item():.2f}, "
                  f"内容损失={content_score.item():.2f}, "
                  f"风格损失={style_score.item():.2f}")
        
        step += 1
        return total_loss
    
    # 优化一步
    optimizer.step(closure)

# ============================================================================
# 第八部分：查看结果!
# ============================================================================

print("\n" + "=" * 60)
print("风格迁移完成!")
print("=" * 60)

# 显示最终结果
with torch.no_grad():
    final_image = input_image.clamp(0, 1).squeeze()
    
    # 转成 numpy 显示
    final_np = final_image.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(content_image.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
    plt.title('原图 (内容)', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(style_image.squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
    plt.title('风格参考', fontsize=12)
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(final_np)
    plt.title('风格迁移结果', fontsize=12)
    plt.axis('off')
    
    plt.suptitle('风格迁移完成!', fontsize=16)
    plt.tight_layout()
    plt.savefig('style_transfer_result.png', dpi=150)
    print("✓ 结果已保存为 'style_transfer_result.png'")
    plt.show()

# 保存生成图片
final_pil = transforms.ToPILImage()(final_image)
final_pil.save('output_stylized.png')
print("✓ 高清结果已保存为 'output_stylized.png'")

# 绘制损失曲线
plt.figure(figsize=(10, 5))
plt.plot(losses, linewidth=2, color='red')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Total Loss', fontsize=12)
plt.title('风格迁移损失变化', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('style_transfer_loss.png', dpi=150)
print("✓ 损失曲线已保存")
plt.show()

print("\n🎉 恭喜！你已经完成了风格迁移!")
print("\n下一步:")
print("  1. 试试不同的风格图片")
print("  2. 调整 style_weight 看效果")
print("  3. 尝试不同的内容层和风格层")
print("  4. 做成手机 APP 或小程序")

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day19](../Day19/README.md)
- [→ Day21](../Day21/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*
