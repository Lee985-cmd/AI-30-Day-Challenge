"""
Day 1: Python 基础示例
包含变量、数据类型、控制流等基础概念
"""

# ============================================================================
# 示例 1: Hello World
# ============================================================================
print("Hello, AI World!")
print("欢迎来到 AI 30天挑战！")

# ============================================================================
# 示例 2: 变量和数据类型
# ============================================================================
# 整数
age = 25
print(f"年龄: {age}, 类型: {type(age)}")

# 浮点数
height = 1.75
print(f"身高: {height}m, 类型: {type(height)}")

# 字符串
name = "AI学习者"
print(f"姓名: {name}, 类型: {type(name)}")

# 布尔值
is_learning = True
print(f"正在学习: {is_learning}, 类型: {type(is_learning)}")

# 列表
skills = ["Python", "Machine Learning", "Deep Learning"]
print(f"技能列表: {skills}")

# 字典
student = {
    "name": "小明",
    "age": 25,
    "major": "Computer Science"
}
print(f"学生信息: {student}")

# ============================================================================
# 示例 3: 条件判断
# ============================================================================
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"

print(f"分数: {score}, 等级: {grade}")

# ============================================================================
# 示例 4: 循环
# ============================================================================
# for 循环
print("\n计数 1-5:")
for i in range(1, 6):
    print(i, end=" ")

# while 循环
print("\n\nWhile 循环:")
count = 0
while count < 5:
    print(count, end=" ")
    count += 1

# ============================================================================
# 示例 5: 函数
# ============================================================================
def greet(name):
    """向某人打招呼"""
    return f"你好, {name}! 欢迎学习 AI!"

message = greet("张三")
print(f"\n\n{message}")

def calculate_bmi(weight, height):
    """计算 BMI 指数"""
    bmi = weight / (height ** 2)
    return round(bmi, 2)

bmi = calculate_bmi(70, 1.75)
print(f"BMI 指数: {bmi}")

# ============================================================================
# 示例 6: 列表推导式
# ============================================================================
# 生成平方数列表
squares = [x**2 for x in range(1, 11)]
print(f"\n1-10 的平方: {squares}")

# 过滤偶数
even_numbers = [x for x in range(1, 21) if x % 2 == 0]
print(f"1-20 的偶数: {even_numbers}")

print("\n✅ Day 1 基础示例完成！")
