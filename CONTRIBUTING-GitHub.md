# 贡献指南 (Contributing Guide)

首先，感谢你考虑为 **AI 入门 30 天挑战** 做出贡献！🎉

本项目致力于建立一个开放、友好、包容的社区。每一份贡献，无论大小，都非常宝贵。

---

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
- [开发环境设置](#开发环境设置)
- [提交 Issue](#提交-issue)
- [提交 Pull Request](#提交-pull-request)
- [代码规范](#代码规范)
- [文档规范](#文档规范)
- [审查流程](#审查流程)
- [常见问题](#常见问题)

---

## 行为准则

本项目采用 [Contributor Covenant](https://www.contributor-covenant.org/) 行为准则。

**简而言之:**
- ✅ 尊重他人，友善交流
- ✅ 建设性批评，对事不对人
- ✅ 接受反馈，持续改进
- ❌ 不人身攻击、歧视或骚扰
- ❌ 不发布不当内容

详细请阅读 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

---

## 如何贡献

### 贡献类型

我们欢迎以下类型的贡献：

#### 1. 🐛 Bug 修复
- 发现并修复代码错误
- 修正文档中的错误信息
- 解决兼容性问题

#### 2. 💡 新功能
- 添加新的代码示例
- 创建新的学习项目
- 改进现有功能

#### 3. 📝 文档改进
- 修正拼写和语法错误
- 补充缺失的说明
- 添加更多示例和图解
- 翻译成其他语言

#### 4. 🎨 代码优化
- 提高代码性能
- 重构代码结构
- 添加单元测试

#### 5. 🌍 本地化
- 翻译成其他语言
- 适配不同地区的内容

---

## 开发环境设置

### 1. Fork 仓库

在 GitHub 上点击 "Fork" 按钮，创建你自己的副本。

### 2. 克隆到本地

```bash
git clone https://github.com/Lee985-cmd/AI-30-Day-Challenge.git
cd AI-30-Day-Challenge
```

### 3. 添加上游远程仓库

```bash
git remote add upstream https://github.com/ORIGINAL-OWNER/AI-30-Day-Challenge.git
```

### 4. 创建虚拟环境

```bash
# Python venv
python -m venv ai-env
source ai-env/bin/activate  # Linux/Mac
# ai-env\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 5. 创建分支

```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b fix/issue-description
```

**分支命名规范:**
- `feature/xxx` - 新功能
- `fix/xxx` - Bug 修复
- `docs/xxx` - 文档更新
- `refactor/xxx` - 代码重构

---

## 提交 Issue

### Bug 报告

使用 [Bug Report](https://github.com/Lee985-cmd/AI-30-Day-Challenge/issues/new?template=bug_report.md) 模板。

**包含:**
- 📝 清晰的标题
- 🔍 问题描述
- 💻 环境信息 (OS, Python版本等)
- 📋 复现步骤
- 🖼️ 截图或错误日志（如有）
- ✅ 预期行为 vs 实际行为

**示例:**
```markdown
**标题:** Day14 CIFAR-10 训练脚本报错

**环境:**
- OS: Windows 11
- Python: 3.9.7
- PyTorch: 2.0.1

**复现步骤:**
1. 运行 code/Day14/cifar10_project.py
2. 训练到第 5 个 epoch
3. 出现 CUDA out of memory 错误

**预期:** 正常完成训练
**实际:** 显存溢出崩溃
```

### 功能建议

使用 [Feature Request](https://github.com/Lee985-cmd/AI-30-Day-Challenge/issues/new?template=feature_request.md) 模板。

**包含:**
- 💡 功能描述
- 🎯 解决的问题
- 📊 优先级评估
- 🔗 相关资源

---

## 提交 Pull Request

### PR 流程

1. **确保代码通过测试**
   ```bash
   python -m pytest tests/
   ```

2. **更新文档**
   - 如果添加了新功能，更新 README
   - 如果修改了 API，更新相关文档

3. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

4. **推送到你的 Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **创建 Pull Request**
   - 前往原仓库
   - 点击 "Compare & pull request"
   - 填写 PR 描述

### PR 模板

```markdown
## 📝 描述
简要描述这个 PR 做了什么

## 🔗 相关 Issue
Closes #123

## 🧪 测试
- [ ] 已添加单元测试
- [ ] 所有测试通过
- [ ] 手动测试过

## 📸 截图 (如适用)
添加前后对比截图

## ✅ 检查清单
- [ ] 代码遵循项目规范
- [ ] 已添加必要的注释
- [ ] 文档已更新
- [ ] 没有破坏性变更
- [ ] 已自我审查代码
```

### PR 最佳实践

✅ **要做:**
- 保持 PR 小而专注（一个 PR 只做一件事）
- 提供清晰的描述和动机
- 回复审查意见
- 及时更新分支（rebase）

❌ **不要:**
- 一次性提交大量无关更改
- 忽略 CI 失败
- 长时间不回应审查意见

---

## 代码规范

### Python 代码

我们遵循 [PEP 8](https://peps.python.org/pep-0008/) 风格指南。

**关键要点:**

```python
# ✅ 好的命名
def calculate_accuracy(predictions, targets):
    """计算准确率"""
    pass

# ❌ 不好的命名
def calc(p, t):
    pass

# ✅ 适当的空格
x = 1 + 2
my_list = [1, 2, 3]

# ❌ 不好的格式
x=1+2
my_list=[1,2,3]

# ✅ 文档字符串
def train_model(model, epochs=10):
    """
    训练模型
    
    Args:
        model: PyTorch 模型
        epochs: 训练轮数
    
    Returns:
        训练历史字典
    """
    pass
```

**工具推荐:**
```bash
# 自动格式化
pip install black
black code/

# 代码检查
pip install flake8
flake8 code/

# 类型检查
pip install mypy
mypy code/
```

### Markdown 文档

```markdown
# 使用清晰的标题层级

**粗体**强调重点  
*斜体*标注术语  
`行内代码`用于代码

```python
# 代码块指定语言
def hello():
    print("Hello!")
```

- 列表项保持一致
- 链接要有描述性文字

[查看文档](link) 而不是 [点击这里](link)
```

---

## 文档规范

### 文件命名

- 使用中文文件名（与教程一致）
- 使用短横线分隔单词
- 避免特殊字符

```
✅ Day01-Q1 - 什么是编程和AI.md
❌ Day01_Q1_什么是编程和AI.md
❌ day01-q1.md
```

### 内容结构

每个 Markdown 文件应包含：

```markdown
# 标题

## 简介
简要说明本节内容

## 核心概念
详细解释...

## 代码示例
```python
# 完整可运行的代码
```

## 常见错误
- 错误1及解决方法
- 错误2及解决方法

## 小结
关键点总结

## 相关链接
- [前一节](link)
- [后一节](link)
```

### 图片处理

- 使用 PNG 格式（图表）或 JPG（照片）
- 压缩图片以减少文件大小
- 添加 alt 文本
- 存放在 `images/` 目录

```markdown
![CNN架构示意图](../images/diagrams/cnn-architecture.png)
```

---

## 审查流程

### 审查标准

维护者会检查：

1. **功能性**
   - 代码是否能正常运行
   - 是否达到预期效果

2. **代码质量**
   - 是否遵循代码规范
   - 是否有清晰的注释
   - 是否有效率问题

3. **文档完整性**
   - 是否更新了相关文档
   - 说明是否清晰

4. **测试覆盖**
   - 是否添加了测试
   - 测试是否通过

### 审查时间

- 简单修复: 1-2 天
- 新功能: 3-5 天
- 大型改动: 1-2 周

### 审查反馈

维护者可能会：
- ✅ 直接合并
- 💬 提出修改建议
- ❌ 拒绝并说明原因

**请耐心对待审查意见，这是提高代码质量的过程。**

---

## 常见问题

### Q: 我是初学者，能贡献吗？

**A:** 当然可以！我们从简单的任务开始：
- 修正拼写错误
- 补充文档说明
- 添加代码注释

查看 [good first issue](https://github.com/Lee985-cmd/AI-30-Day-Challenge/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) 标签。

### Q: 我的 PR 多久会被审查？

**A:** 通常 1-3 天内会有回应。如果超过一周没有回复，可以 @ 维护者。

### Q: 如何同步上游的最新更改？

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Q: 可以同时开多个 PR 吗？

**A:** 可以，但建议：
- 每个 PR 专注于一个功能
- 避免 PR 之间冲突
- 先完成一个再开下一个

### Q: 我的代码被拒绝了怎么办？

**A:** 不要灰心！
- 仔细阅读反馈意见
- 理解拒绝的原因
- 修改后重新提交
- 或者开 Issue 讨论

---

## 认可贡献者

所有贡献者都会被记录在 [CONTRIBUTORS.md](CONTRIBUTORS.md) 文件中。

**贡献等级:**
- 🥇 **Core Contributor**: 10+ PRs 合并
- 🥈 **Active Contributor**: 5-9 PRs 合并
- 🥉 **Contributor**: 1-4 PRs 合并

---

## 联系方式

有问题或建议？

- 📧 Email: maintainers@example.com
- 💬 Discord: [加入社区](https://discord.gg/invite)
- 🐛 Issues: [GitHub Issues](https://github.com/Lee985-cmd/AI-30-Day-Challenge/issues)

---

## 许可证

通过贡献代码，你同意你的贡献将在 [MIT License](LICENSE) 下发布。

---

**再次感谢你的贡献！** 🎉

每一行代码、每一个字，都在帮助更多人学习 AI。

**一起让这个项目变得更好！** 💪✨
