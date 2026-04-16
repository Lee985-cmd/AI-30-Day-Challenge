# 🔐 API Key 安全配置指南

## ⚠️ 重要提醒

**绝对不要将 API Key 硬编码在代码中或提交到 Git！**

---

## ✅ 推荐方案：系统环境变量

### **Windows PowerShell**

#### **临时设置（当前窗口有效）**

```powershell
$env:DASHSCOPE_API_KEY="sk-your-actual-api-key-here"
```

**优点：**
- ✅ 简单快速
- ✅ 关闭窗口后自动清除

**缺点：**
- ❌ 每次打开新窗口都需要重新设置

---

#### **永久设置（推荐）**

```powershell
# 设置用户级别的环境变量
[System.Environment]::SetEnvironmentVariable("DASHSCOPE_API_KEY", "sk-your-actual-api-key-here", "User")
```

**验证是否设置成功：**
```powershell
# 关闭并重新打开 PowerShell
echo $env:DASHSCOPE_API_KEY
```

**优点：**
- ✅ 一次设置，永久有效
- ✅ 所有程序都可以访问
- ✅ 不会出现在代码中

**缺点：**
- ❌ 需要管理员权限（系统级别）

---

### **Windows CMD**

#### **临时设置**

```cmd
set DASHSCOPE_API_KEY=sk-your-actual-api-key-here
```

#### **永久设置**

```cmd
setx DASHSCOPE_API_KEY "sk-your-actual-api-key-here"
```

> ⚠️ 注意：`setx` 设置后需要重新打开命令行窗口才生效

---

### **Linux/Mac**

#### **临时设置**

```bash
export DASHSCOPE_API_KEY="sk-your-actual-api-key-here"
```

#### **永久设置**

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export DASHSCOPE_API_KEY="sk-your-actual-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

---

## 🚀 使用启动脚本（最方便）

### **Windows**

双击运行以下任一脚本：

- `start.bat` - CMD 版本
- `start.ps1` - PowerShell 版本

脚本会自动检查 API Key 是否设置，如果未设置会给出提示。

---

## ❌ 不推荐的方案

### **1. .env 文件**

```python
# 不推荐！容易泄露
DASHSCOPE_API_KEY=sk-your-key
```

**风险：**
- ❌ 可能忘记添加到 `.gitignore`
- ❌ 可能被上传到公开仓库
- ❌ 多人协作时难以管理

---

### **2. 硬编码在代码中**

```python
# 绝对禁止！
api_key = "sk-your-actual-api-key-here"
```

**风险：**
- ❌ 直接暴露在代码中
- ❌ 提交到 Git 后无法彻底删除
- ❌ 任何人都可以看到

---

### **3. 配置文件**

```python
# config.py
API_KEY = "sk-your-key"
```

**风险：**
- ❌ 同 .env 文件
- ❌ 容易被导入和泄露

---

## 🔍 检查 API Key 是否泄露

### **1. 检查 Git 历史**

```bash
# 搜索是否曾经提交过 API Key
git log --all -p -S "sk-" --oneline
```

如果发现泄露，立即：
1. 撤销该 API Key
2. 创建新的 API Key
3. 使用 `git filter-branch` 清理历史

---

### **2. 检查当前代码**

```bash
# 搜索代码中是否包含 API Key
grep -r "sk-" projects/ai-customer-service/
```

---

## 🛡️ 最佳实践

### **1. 定期轮换 API Key**

- 每 3-6 个月更换一次
- 发现异常使用时立即更换

---

### **2. 最小权限原则**

- 只为需要的服务开通权限
- 设置使用额度限制

---

### **3. 监控使用情况**

定期查看阿里云控制台的使用记录：
https://dashscope.console.aliyun.com/usage

---

### **4. 团队协作**

**不要共享 API Key！**

每个开发者应该：
1. 注册自己的阿里云账号
2. 获取自己的 API Key
3. 独立配置环境变量

---

## 📋 安全检查清单

启动项目前，确认：

- [ ] API Key 已设置为环境变量
- [ ] 代码中没有硬编码的 API Key
- [ ] `.env` 文件已添加到 `.gitignore`
- [ ] 没有将 API Key 提交到 Git
- [ ] 使用启动脚本检查环境变量

---

## 🆘 如果 API Key 泄露了

### **立即执行：**

1. **撤销 API Key**
   - 登录阿里云控制台
   - 进入 API-KEY 管理
   - 删除泄露的 Key

2. **创建新 Key**
   - 生成新的 API Key
   - 更新本地环境变量

3. **检查使用记录**
   - 查看是否有异常调用
   - 联系阿里云客服（如有损失）

4. **清理 Git 历史**（如果已提交）
   ```bash
   # 警告：这会重写 Git 历史
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch projects/ai-customer-service/.env' \
     --prune-empty --tag-name-filter cat -- --all
   ```

---

## 💡 总结

| 方案 | 安全性 | 便利性 | 推荐度 |
|------|--------|--------|--------|
| 系统环境变量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅✅✅ 强烈推荐 |
| 启动脚本 + 环境变量 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅✅✅ 最推荐 |
| .env 文件 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ 谨慎使用 |
| 硬编码 | ⭐ | ⭐⭐⭐⭐⭐ | ❌ 绝对禁止 |

**最佳选择：系统环境变量 + 启动脚本**

---

**保护你的 API Key，就是保护你的钱包！** 💰🔒
