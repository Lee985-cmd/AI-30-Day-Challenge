# 🔒 安全配置指南

## ⚠️ 重要安全原则

**永远不要在代码或文档中暴露以下敏感信息：**

1. ❌ **真实的服务器地址和端口**
2. ❌ **API Keys 和访问令牌**
3. ❌ **数据库连接字符串**
4. ❌ **用户名和密码**
5. ❌ **内部网络IP地址**

---

## ✅ 正确的配置方式

### 1. 使用环境变量

**错误做法：**
```python
# ❌ 硬编码URL（危险！）
LOCAL_LLM_URL = "http://61.49.53.5:30001/v1"
```

**正确做法：**
```python
# ✅ 从环境变量读取
import os
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL")
if not LOCAL_LLM_URL:
    raise ValueError("请设置 LOCAL_LLM_URL 环境变量")
```

### 2. 文档中使用通用示例

**错误做法：**
```markdown
配置示例：`http://61.49.53.5:30001/v1`
```

**正确做法：**
```markdown
配置示例：`http://localhost:8000/v1` 或 `http://your-server:port/v1`
```

### 3. .gitignore 保护敏感文件

确保以下文件被加入 `.gitignore`：

```gitignore
# 环境变量文件
.env
.env.local
*.env

# 配置文件（可能包含敏感信息）
config.yaml
settings.json

# 日志文件（可能泄露信息）
*.log
logs/
```

---

## 🛡️ 本项目安全措施

### 已实施的安全措施

1. ✅ **使用环境变量**
   - `LOCAL_LLM_URL` 通过环境变量配置
   - 不在代码中硬编码任何URL

2. ✅ **文档脱敏**
   - README.md 使用通用示例
   - QUICKSTART.md 不显示真实地址
   - 所有文档避免暴露内网IP

3. ✅ **.gitignore 配置**
   ```gitignore
   # 环境变量
   .env
   *.env
   
   # 推广内容（不公开）
   promotion-articles/
   
   # 生成的报告
   reports/
   ```

4. ✅ **代码审查**
   - 提交前检查是否有敏感信息
   - 使用占位符代替真实值

### 如何验证安全性

运行以下命令检查是否有敏感信息泄露：

```bash
# 检查代码中是否有IP地址
grep -r "[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}" projects/

# 检查是否有API Key
grep -r "sk-" projects/
grep -r "api_key" projects/

# 检查是否有密码
grep -r "password" projects/
grep -r "passwd" projects/
```

---

## 📝 配置本地模型（用户操作指南）

### Windows PowerShell

```powershell
# 临时设置（当前会话有效）
$env:LOCAL_LLM_URL = "http://your-server:port/v1"

# 永久设置（需要管理员权限）
[System.Environment]::SetEnvironmentVariable("LOCAL_LLM_URL", "http://your-server:port/v1", "User")
```

### Linux/Mac

```bash
# 临时设置
export LOCAL_LLM_URL="http://your-server:port/v1"

# 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
echo 'export LOCAL_LLM_URL="http://your-server:port/v1"' >> ~/.bashrc
source ~/.bashrc
```

### 验证配置

```bash
# Windows PowerShell
echo $env:LOCAL_LLM_URL

# Linux/Mac
echo $LOCAL_LLM_URL
```

---

## 🔍 常见安全问题

### Q1: 我不小心提交了敏感信息怎么办？

**立即采取以下措施：**

1. **撤销提交**（如果还未推送）
   ```bash
   git reset --soft HEAD~1
   # 修改文件后重新提交
   ```

2. **如果已推送到GitHub**
   - 立即更改密码/API Key
   - 联系GitHub支持删除历史记录
   - 使用 `git filter-branch` 清理历史

3. **预防措施**
   - 使用 pre-commit hooks 自动检查
   - 启用GitHub的secret scanning功能

### Q2: 如何在团队中安全共享配置？

**推荐方案：**

1. **提供配置模板**
   ```bash
   # .env.example（可以提交到Git）
   LOCAL_LLM_URL=http://localhost:8000/v1
   ```

2. **团队成员各自创建 .env**
   ```bash
   # .env（不要提交到Git）
   LOCAL_LLM_URL=http://your-actual-server:port/v1
   ```

3. **使用配置管理工具**
   - HashiCorp Vault
   - AWS Secrets Manager
   - Azure Key Vault

### Q3: 开源项目如何处理敏感配置？

**最佳实践：**

1. ✅ 使用环境变量
2. ✅ 提供 `.env.example` 模板
3. ✅ 在README中说明配置方法
4. ✅ 使用占位符和通用示例
5. ❌ 不要提交真实的配置文件
6. ❌ 不要在文档中暴露内网地址

---

## 🎯 安全检查清单

在每次提交代码前，确认：

- [ ] 没有硬编码的IP地址或域名
- [ ] 没有API Keys或访问令牌
- [ ] 没有用户名或密码
- [ ] 没有数据库连接字符串
- [ ] 使用了环境变量或配置文件
- [ ] 敏感文件已加入 `.gitignore`
- [ ] 文档中使用通用示例而非真实值
- [ ] 注释中没有敏感信息

---

## 📚 相关资源

- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [OWASP Secure Coding Practices](https://owasp.org/www-project-secure-coding-practices-quick-reference-guide/)
- [12-Factor App: Config](https://12factor.net/config)

---

**记住：安全无小事，谨慎处理每一个敏感信息！** 🔐
