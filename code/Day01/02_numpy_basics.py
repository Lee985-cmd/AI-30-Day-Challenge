"""
Day 2: NumPy 基础
数组操作、数学运算、数据处理
"""

import numpy as np

print("=" * 60)
print("NumPy 基础教程")
print("=" * 60)

# ============================================================================
# 示例 1: 创建数组
# ============================================================================
print("\n【1. 创建数组】")

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {arr1}")

# 二维数组
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组:\n{arr2}")

# 特殊数组
zeros = np.zeros((3, 4))
print(f"\n零数组 (3x4):\n{zeros}")

ones = np.ones((2, 3))
print(f"\n全1数组 (2x3):\n{ones}")

identity = np.eye(3)
print(f"\n单位矩阵 (3x3):\n{identity}")

# 范围数组
range_arr = np.arange(0, 10, 2)
print(f"\n范围数组 (0-10, 步长2): {range_arr}")

# 等间距数组
linspace_arr = np.linspace(0, 1, 5)
print(f"等间距数组 (0-1, 5个点): {linspace_arr}")

# ============================================================================
# 示例 2: 数组属性
# ============================================================================
print("\n【2. 数组属性】")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"数组:\n{arr}")
print(f"维度 (ndim): {arr.ndim}")
print(f"形状 (shape): {arr.shape}")
print(f"元素总数 (size): {arr.size}")
print(f"数据类型 (dtype): {arr.dtype}")

# ============================================================================
# 示例 3: 数组索引和切片
# ============================================================================
print("\n【3. 索引和切片】")

arr = np.array([10, 20, 30, 40, 50])
print(f"原始数组: {arr}")
print(f"第一个元素: {arr[0]}")
print(f"最后一个元素: {arr[-1]}")
print(f"前三个元素: {arr[:3]}")
print(f"最后两个元素: {arr[-2:]}")

# 二维数组索引
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n二维数组:\n{arr_2d}")
print(f"第2行第3列: {arr_2d[1, 2]}")
print(f"第2行: {arr_2d[1, :]}")
print(f"第3列: {arr_2d[:, 2]}")

# ============================================================================
# 示例 4: 数组运算
# ============================================================================
print("\n【4. 数组运算】")

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")

# 标量运算
print(f"\na + 10 = {a + 10}")
print(f"a * 2 = {a * 2}")

# ============================================================================
# 示例 5: 数学函数
# ============================================================================
print("\n【5. 数学函数】")

arr = np.array([1, 4, 9, 16, 25])
print(f"原始数组: {arr}")
print(f"平方根: {np.sqrt(arr)}")
print(f"自然对数: {np.log(arr)}")
print(f"指数: {np.exp(arr[:3])}")  # 只显示前3个，避免数值过大

# 三角函数
angles = np.array([0, np.pi/2, np.pi])
print(f"\n角度: {angles}")
print(f"sin: {np.sin(angles)}")
print(f"cos: {np.cos(angles)}")

# ============================================================================
# 示例 6: 统计函数
# ============================================================================
print("\n【6. 统计函数】")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"数组: {arr}")
print(f"总和: {np.sum(arr)}")
print(f"平均值: {np.mean(arr)}")
print(f"中位数: {np.median(arr)}")
print(f"标准差: {np.std(arr):.2f}")
print(f"方差: {np.var(arr):.2f}")
print(f"最小值: {np.min(arr)}")
print(f"最大值: {np.max(arr)}")
print(f"最小值索引: {np.argmin(arr)}")
print(f"最大值索引: {np.argmax(arr)}")

# ============================================================================
# 示例 7: 数组重塑
# ============================================================================
print("\n【7. 数组重塑】")

arr = np.arange(1, 13)
print(f"原始数组 (12个元素): {arr}")

# 重塑为 3x4
reshaped = arr.reshape(3, 4)
print(f"\n重塑为 3x4:\n{reshaped}")

# 重塑为 2x6
reshaped2 = arr.reshape(2, 6)
print(f"\n重塑为 2x6:\n{reshaped2}")

# 展平
flattened = reshaped.flatten()
print(f"\n展平后: {flattened}")

# ============================================================================
# 示例 8: 数组拼接和分割
# ============================================================================
print("\n【8. 拼接和分割】")

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print(f"数组 a:\n{a}")
print(f"数组 b:\n{b}")

# 垂直拼接
v_stack = np.vstack((a, b))
print(f"\n垂直拼接:\n{v_stack}")

# 水平拼接
h_stack = np.hstack((a, b))
print(f"\n水平拼接:\n{h_stack}")

# 分割
arr = np.arange(1, 13)
split_arr = np.array_split(arr, 3)
print(f"\n分割为3部分:")
for i, part in enumerate(split_arr):
    print(f"  部分 {i+1}: {part}")

# ============================================================================
# 示例 9: 布尔索引
# ============================================================================
print("\n【9. 布尔索引】")

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"原始数组: {arr}")

# 筛选大于5的元素
greater_than_5 = arr[arr > 5]
print(f"大于5的元素: {greater_than_5}")

# 筛选偶数
even_numbers = arr[arr % 2 == 0]
print(f"偶数: {even_numbers}")

# 多条件
condition = (arr > 3) & (arr < 8)
filtered = arr[condition]
print(f"大于3且小于8的元素: {filtered}")

# ============================================================================
# 示例 10: 随机数生成
# ============================================================================
print("\n【10. 随机数生成】")

# 设置随机种子（保证可重复）
np.random.seed(42)

# 均匀分布
random_uniform = np.random.rand(3, 3)
print(f"均匀分布随机数 (3x3):\n{random_uniform}")

# 正态分布
random_normal = np.random.randn(3, 3)
print(f"\n正态分布随机数 (3x3):\n{random_normal}")

# 整数随机数
random_int = np.random.randint(1, 100, size=10)
print(f"\n1-100的随机整数 (10个): {random_int}")

# 随机选择
choices = np.random.choice(['A', 'B', 'C', 'D'], size=5)
print(f"随机选择: {choices}")

print("\n" + "=" * 60)
print("✅ NumPy 基础教程完成！")
print("=" * 60)
