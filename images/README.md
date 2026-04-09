# 图片资源说明

## 📁 目录结构

```
images/
├── diagrams/           # 架构图和流程图
├── screenshots/        # 运行截图
└── logos/             # Logo 和品牌图片
```

## 🎨 需要的图片清单

### Diagrams (架构图)

#### 1. 神经网络架构
- **文件名**: `neural-network-architecture.png`
- **尺寸**: 1200x800
- **内容**: 展示输入层、隐藏层、输出层的连接
- **工具推荐**: draw.io, Lucidchart, Excalidraw

#### 2. CNN 架构
- **文件名**: `cnn-architecture.png`
- **尺寸**: 1400x900
- **内容**: 卷积层、池化层、全连接层的流程
- **包含**: 特征图可视化

#### 3. Transformer 架构
- **文件名**: `transformer-architecture.png`
- **尺寸**: 1600x1000
- **内容**: Encoder-Decoder、Attention 机制
- **参考**: "Attention Is All You Need" 论文图

#### 4. YOLO 检测流程
- **文件名**: `yolo-detection-pipeline.png`
- **尺寸**: 1200x800
- **内容**: 图像输入 → 网格划分 → 边界框预测

#### 5. LSTM 单元
- **文件名**: `lstm-cell.png`
- **尺寸**: 1000x600
- **内容**: 遗忘门、输入门、输出门

#### 6. 学习路线图
- **文件名**: `learning-roadmap.png`
- **尺寸**: 1920x1080
- **内容**: 30天学习的完整路径
- **风格**: 时间线或流程图

### Screenshots (运行截图)

#### 1. 训练曲线
- **文件名**: `training-curves.png`
- **内容**: Loss 和 Accuracy 随 epoch 的变化
- **要求**: 清晰的标签和图例

#### 2. 混淆矩阵
- **文件名**: `confusion-matrix.png`
- **内容**: 分类结果的混淆矩阵热力图
- **颜色**: 使用渐变色

#### 3. 预测示例
- **文件名**: `prediction-examples.png`
- **内容**: 多张图片的预测结果对比
- **标注**: 绿色=正确，红色=错误

#### 4. 目标检测结果
- **文件名**: `object-detection-result.png`
- **内容**: 带边界框的检测结果
- **要求**: 清晰的类别标签和置信度

#### 5. 文本生成示例
- **文件名**: `text-generation-example.png`
- **内容**: 生成的文本展示
- **格式**: 代码块或终端输出样式

#### 6. Web API 界面
- **文件名**: `api-docs-screenshot.png`
- **内容**: FastAPI Swagger UI 或 Flask 界面
- **要求**: 展示 API 端点和测试功能

### Logos (品牌图片)

#### 1. 项目 Logo
- **文件名**: `logo.png` (PNG, 透明背景)
- **尺寸**: 512x512, 256x256, 128x128
- **设计元素**: AI、学习、火箭等
- **颜色**: 紫色渐变 (#667eea to #764ba2)

#### 2. Favicon
- **文件名**: `favicon.ico`, `favicon.png`
- **尺寸**: 32x32, 16x16
- **内容**: 简化的 Logo

#### 3. Open Graph 图片
- **文件名**: `og-image.png`
- **尺寸**: 1200x630
- **内容**: 项目标题 + Logo + 简短描述
- **用途**: 社交媒体分享时显示

#### 4. Banner
- **文件名**: `github-banner.png`
- **尺寸**: 1280x640
- **内容**: GitHub README 顶部大图
- **包含**: 项目名称、特色、Star 按钮

## 🛠️ 创建工具推荐

### 免费工具

1. **Excalidraw** (https://excalidraw.com/)
   - 手绘风格的图表
   - 适合架构图
   - 导出 PNG/SVG

2. **draw.io / diagrams.net**
   - 专业的流程图工具
   - 丰富的模板
   - 支持协作

3. **Canva** (https://canva.com/)
   - 设计 Banner 和 Logo
   - 大量模板
   - 易于使用

4. **GIMP** (https://www.gimp.org/)
   - 免费的 Photoshop 替代品
   - 图片编辑和处理

5. **Inkscape** (https://inkscape.org/)
   - 矢量图形编辑
   - 适合 Logo 设计

### 在线工具

1. **Carbon** (https://carbon.now.sh/)
   - 美化代码截图
   - 多种主题

2. **Shields.io** (https://shields.io/)
   - 生成 Badge
   - 用于 README

3. **Star History** (https://star-history.com/)
   - GitHub Star 历史图

## 📐 设计规范

### 颜色方案

**主色调:**
- 紫色: #667eea
- 深紫: #764ba2
- 蓝色: #4facfe
- 青色: #00f2fe

**辅助色:**
- 成功绿: #27c93f
- 警告黄: #ffbd2e
- 错误红: #ff5f56
- 中性灰: #6c757d

### 字体

- **标题**: Inter, Roboto, Arial
- **正文**: Inter, system-ui
- **代码**: Fira Code, Consolas, Monaco

### 风格指南

1. **一致性**
   - 所有图表使用相同的颜色方案
   - 统一的字体大小和样式
   - 相似的边框和阴影效果

2. **清晰度**
   - 文字足够大（至少 12pt）
   - 足够的对比度
   - 避免过度拥挤

3. **专业性**
   - 简洁的设计
   - 适当的留白
   - 对齐和间距一致

## 📝 图片优化

### 压缩工具

1. **TinyPNG** (https://tinypng.com/)
   - 智能压缩 PNG/JPG
   - 保持质量的同时减小文件大小

2. **ImageOptim** (Mac)
   - 批量优化图片
   - 移除元数据

3. **Squoosh** (https://squoosh.app/)
   - Google 开发的在线工具
   - 实时预览压缩效果

### 优化建议

- PNG: 用于图表、Logo（需要透明背景）
- JPG: 用于照片、截图
- SVG: 用于简单图标、Logo（可缩放）
- WebP: 现代格式，更好的压缩率

**目标文件大小:**
- 小图标: < 10 KB
- 普通图片: < 100 KB
- Banner: < 300 KB

## 🎯 优先级

### 高优先级（必须）
1. ✅ GitHub Banner
2. ✅ Logo 和 Favicon
3. ✅ 学习路线图
4. ✅ 神经网络架构图
5. ✅ 训练曲线示例

### 中优先级（推荐）
6. CNN/RNN/Transformer 架构图
7. 预测结果截图
8. API 文档截图
9. Open Graph 图片

### 低优先级（可选）
10. 更多示例截图
11. 视频教程缩略图
12. 社区活动图片

## 🔗 免费图片资源

如果需要通用图片（非定制图表）：

1. **Unsplash** (https://unsplash.com/)
   - 高质量免费照片
   - 可商用

2. **Pexels** (https://www.pexels.com/)
   - 免费 stock photos
   - 无需署名

3. **Pixabay** (https://pixabay.com/)
   - 图片、矢量图、视频
   - 免费使用

## 💡 提示

1. **保持一致性**: 所有图片使用相同的风格和配色
2. **添加 Alt 文本**: 提高可访问性
3. **响应式**: 确保在不同设备上显示良好
4. ** lazy loading**: 大图片使用懒加载
5. **CDN**: 考虑使用 CDN 加速图片加载

---

**当前状态:** 📋 待创建  
**预计完成时间:** 1-2 天  
**负责人:** 社区贡献者欢迎参与！
