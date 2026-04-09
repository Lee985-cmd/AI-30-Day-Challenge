# Day 01 - Python 和 NumPy 基础

## 📁 代码文件说明

本目录包含 Day 1 的所有代码示例。

### 文件列表

- `01_python_basics.py` - Python 基础语法示例
  - Hello World
  - 变量和数据类型
  - 条件判断
  - 循环
  - 函数
  - 列表推导式

- `02_numpy_basics.py` - NumPy 数组操作
  - 创建数组
  - 数组属性
  - 索引和切片
  - 数组运算
  - 数学函数
  - 统计函数
  - 数组重塑
  - 拼接和分割
  - 布尔索引
  - 随机数生成

## 🚀 运行方法

### 前置要求

确保已安装 Python 3.7+ 和必要的库：

```bash
pip install numpy
```

### 运行代码

```bash
# 运行 Python 基础示例
python 01_python_basics.py

# 运行 NumPy 示例
python 02_numpy_basics.py
```

## 📝 学习要点

### Python 基础
- 变量命名规范
- 数据类型的区别和使用场景
- 控制流的最佳实践
- 函数定义和调用

### NumPy 核心
- 为什么使用 NumPy 而不是原生列表？
  - 更快的运算速度
  - 更少的内存占用
  - 丰富的数学函数
  
- 数组 vs 列表
  - 数组：同类型元素，支持向量化运算
  - 列表：可混合类型，更灵活但较慢

## 💡 常见错误

### 错误 1: 形状不匹配

```python
# ❌ 错误
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
c = a + b  # ValueError!

# ✅ 正确：确保形状兼容
a = np.array([1, 2])
b = np.array([[1, 2], [3, 4]])
c = a + b  # 广播机制
```

### 错误 2: 原地修改

```python
# ❌ 可能意外修改原数组
a = np.array([1, 2, 3])
b = a
b[0] = 100  # a 也被修改了！

# ✅ 使用 copy
a = np.array([1, 2, 3])
b = a.copy()
b[0] = 100  # a 不受影响
```

## 🔗 相关链接

- [← 回到 Day01 文档](../../Day01/README.md)
- [→ Day02 代码](../Day02/)
- [NumPy 官方文档](https://numpy.org/doc/)
