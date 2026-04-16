# Markdown 转公众号工具使用说明

## 📖 功能介绍

将 Markdown 格式的文章自动转换为适合微信公众号的富文本格式，解决公众号编辑器格式混乱的问题。

---

## 🚀 快速开始

### 1. 转换文章

```bash
# 基本用法
python tools/md_to_wechat.py promotion-articles/公众号首篇.md

# 指定输出文件
python tools/md_to_wechat.py promotion-articles/公众号首篇.md output.html
```

### 2. 粘贴到公众号

1. 用浏览器打开生成的 `.html` 文件
2. 全选复制（Ctrl+A, Ctrl+C）
3. 粘贴到公众号编辑器（Ctrl+V）
4. 完成！✅

---

## ✨ 支持的格式

### ✅ 已支持

- **标题**：`#` `##` `###` → 不同层级的标题样式
- **加粗**：`**text**` → 加粗显示
- **斜体**：`*text*` → 斜体显示
- **列表**：`- item` → 无序列表
- **引用**：`> text` → 引用块样式
- **代码块**：\`\`\`code\`\`\` → 深色背景代码块
- **分隔线**：`---` → 水平分割线
- **链接**：`[text](url)` → 可点击链接
- **段落**：自动优化行距和间距

### 🎨 样式特点

- 标题层次分明（18px/16px/15px）
- 代码块深色主题（类似 VS Code）
- 引用块左侧蓝色边框
- 合理的行距（1.8）和字号（15px）
- 移动端友好（最大宽度 677px）

---

## 📝 使用示例

### 示例 1：转换首篇文章

```bash
cd "e:\learn\AI 入门 30 天挑战"
python tools/md_to_wechat.py promotion-articles/公众号首篇.md
```

生成文件：`promotion-articles/公众号首篇_wechat.html`

### 示例 2：批量转换

创建批处理脚本 `batch_convert.bat`：

```batch
@echo off
echo 开始批量转换...

python tools/md_to_wechat.py promotion-articles/公众号首篇.md
python tools/md_to_wechat.py promotion-articles/公众号文章-AI 客服系统实战.md

echo 转换完成！
pause
```

---

## 💡 高级用法

### 自定义样式

修改 `md_to_wechat.py` 中的样式配置：

```python
# 标题样式
f'<strong style="font-size: 18px; color: #2c3e50;">{line[2:]}</strong>'

# 代码块背景色
background: #282c34

# 正文字号和行距
font-size: 15px; line-height: 1.8
```

### 添加更多格式支持

在 `convert()` 方法中添加新的处理逻辑：

```python
# 例如：处理图片
elif line.startswith('!['):
    # 提取图片路径和描述
    result.append(f'<img src="..." alt="...">')
```

---

## 🔧 常见问题

### Q1: 为什么不用 Markdown Nice？

A: 本地工具的优势：
- ✅ 无需联网
- ✅ 完全可控
- ✅ 可定制样式
- ✅ 批量处理方便

### Q2: 转换后格式不对怎么办？

A: 检查以下几点：
1. Markdown 语法是否规范
2. 是否有特殊字符未转义
3. 代码块是否正确闭合

### Q3: 如何调整样式？

A: 直接修改 `md_to_wechat.py` 中的 HTML 模板，然后重新转换。

---

## 📊 效果对比

### 转换前（Markdown）
```markdown
## 二、30 天挑战

这 30 天里，我完成了：
- Week 1：Python 基础
- Week 2：神经网络
```

### 转换后（公众号格式）
```html
<section style="margin: 15px 0;">
  <strong style="font-size: 16px; color: #34495e;">二、30 天挑战</strong>
</section>
<p style="margin: 10px 0; line-height: 1.8; font-size: 15px; color: #333;">
  这 30 天里，我完成了：
</p>
<section style="margin: 8px 0; padding-left: 20px;">
  <span style="color: #2c3e50; font-size: 15px;">• Week 1：Python 基础</span>
</section>
```

---

## 🎯 最佳实践

1. **写完文章后**：立即运行转换工具
2. **预览检查**：用浏览器打开 HTML 文件检查效果
3. **微调优化**：如需调整，修改 Markdown 源文件后重新转换
4. **备份源文件**：保留 `.md` 文件，方便后续修改

---

## 🚀 未来计划

- [ ] 支持图片自动上传
- [ ] 支持更多主题风格
- [ ] 支持表格格式化
- [ ] 支持数学公式
- [ ] GUI 界面版本

---

**作者**：Lee  
**最后更新**：2026-04-16
