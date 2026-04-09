"""
Day01 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day01_examples.py

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
print("Day01 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

print("你好，世界！")
print("这是我的第一个 Python 程序！")
print(1 + 1)
print("我今年", 25, "岁")

# ===== 代码块 2 =====

print("你好，世界！")
# print = 打印、显示
# "你好，世界！" = 要显示的文字（要用引号括起来）
# 这行的意思：在屏幕上显示"你好，世界！"

print("这是我的第一个 Python 程序！")
# 同样，显示这句话

print(1 + 1)
# 计算 1+1，然后显示结果
# 输出会是：2

print("我今年", 25, "岁")
# 显示多个内容，用逗号隔开
# 输出会是：我今年 25 岁

# ===== 代码块 3 =====

# 这是我的第一行代码
print("Hello, AI!")
print("我正在学习 Python")
print(100 + 200)

# ===== 代码块 4 =====

# 创建一个盒子，标签是 age，里面放数字 25
age = 25

# 创建一个盒子，标签是 name，里面放文字
name = "小明"

# 创建一个盒子，标签是 height，里面放小数
height = 1.75

# 使用这些盒子
print(age)      # 输出：25
print(name)     # 输出：小明
print(height)   # 输出：1.75

# ===== 代码块 5 =====

age = 25
# age = 变量的名字（盒子的标签）
# = = 赋值符号（把右边的东西放进左边的盒子）
# 25 = 数据（盒子里的东西）
# 整句话：创建一个叫 age 的盒子，里面放数字 25

# ===== 代码块 6 =====

# 我的个人信息
my_name = "张三"    # 姓名
my_age = 25         # 年龄
my_height = 1.75    # 身高（米）
my_weight = 65.5    # 体重（公斤）
is_student = True   # 是不是学生

print("姓名：", my_name)
print("年龄：", my_age, "岁")
print("身高：", my_height, "米")
print("体重：", my_weight, "公斤")
print("是学生吗？", is_student)

# ===== 代码块 7 =====

age = 25          # 年龄
count = 100       # 数量
price = 50        # 价格（元）
negative = -10    # 负数

print(age)        # 输出：25
print(type(age))  # 看看这是什么类型
                  # 输出：<class 'int'>

# ===== 代码块 8 =====

height = 1.75     # 身高（米）
weight = 65.5     # 体重（公斤）
score = 95.5      # 分数
pi = 3.14159      # 圆周率

print(height)     # 输出：1.75
print(type(height))  # 输出：<class 'float'>

# ===== 代码块 9 =====

name = "张三"           # 名字
city = '北京'           # 城市
message = "你好世界"    # 消息
empty = ""              # 空字符串

print(name)         # 输出：张三
print(type(name))   # 输出：<class 'str'>

# ===== 代码块 10 =====

is_student = True     # 是学生
is_raining = False    # 没下雨
has_money = True      # 有钱
is_adult = False      # 不是成年人

print(is_student)     # 输出：True
print(type(is_student))  # 输出：<class 'bool'>

# ===== 代码块 11 =====

# 创建各种类型的数据
my_age = 25              # 整数
my_height = 1.75         # 小数
my_name = "李四"         # 文字
is_adult = True          # 对错

# 打印出来，看看类型
print("年龄：", my_age, "类型：", type(my_age))
print("身高：", my_height, "类型：", type(my_height))
print("姓名：", my_name, "类型：", type(my_name))
print("成年：", is_adult, "类型：", type(is_adult))

# ===== 代码块 12 =====

# 创建一个列表
hobbies = ["打游戏", "看电影", "读书", "运动"]

# 访问第 1 个（注意：从 0 开始数！）
print(hobbies[0])   # 输出：打游戏

# 访问第 2 个
print(hobbies[1])   # 输出：看电影

# 访问第 3 个
print(hobbies[2])   # 输出：读书

# 修改第 3 个
hobbies[2] = "旅游"
print(hobbies)      # 输出：['打游戏', '看电影', '旅游', '运动']

# 添加一个新的到最后
hobbies.append("做饭")
print(hobbies)      # 输出：['打游戏', '看电影', '旅游', '运动', '做饭']

# ===== 代码块 13 =====

hobbies = ["打游戏", "看电影", "读书", "运动"]
# hobbies = 列表的名字
# [] = 列表的标志
# [] 里的内容 = 列表的元素
# 每个元素用引号括起来（因为是文字）
# 元素之间用逗号隔开

# ===== 代码块 14 =====

# 你喜欢的东西
favorites = ["pizza", "可乐", "篮球", "周杰伦"]

print("我最喜欢的食物：", favorites[0])
print("我最喜欢的饮料：", favorites[1])
print("我最喜欢的运动：", favorites[2])
print("我最喜欢的歌手：", favorites[3])

# 添加一个新的
favorites.append("Python")
print("\n现在我最喜欢：", favorites)

# 列表的长度（有多少个元素）
print("\n一共有", len(favorites), "个最喜欢的东西")

# ===== 代码块 15 =====

# 创建一个人的信息
person = {
    "name": "张三",
    "age": 25,
    "city": "北京",
    "job": "工程师"
}

# 获取姓名
print(person["name"])   # 输出：张三

# 获取年龄
print(person["age"])    # 输出：25

# 添加新信息
person["salary"] = 10000
print(person)
# 输出：{'name': '张三', 'age': 25, 'city': '北京', 
#       'job': '工程师', 'salary': 10000}

# 修改信息
person["age"] = 26
print(person["age"])    # 输出：26

# ===== 代码块 16 =====

person = {
    "name": "张三",
    # "name" = 键（key），就是标签
    # "张三" = 值（value），就是内容
    # : = 冒号，连接键和值
}
# {} = 字典的标志

# ===== 代码块 17 =====

# 创建你自己的信息
me = {
    "name": "你的名字",
    "age": 你的年龄，
    "hobby": "你的爱好",
    "dream": "你的梦想"
}

print("我叫：", me["name"])
print("今年：", me["age"], "岁")
print("爱好：", me["hobby"])
print("梦想：", me["dream"])

# 添加新信息
me["favorite_color"] = "蓝色"
print("\n我的完整信息：", me)

# ===== 代码块 18 =====

score = 85  # 考试分数

if score >= 90:
    print("优秀！")
elif score >= 80:
    print("良好！")
elif score >= 60:
    print("及格！")
else:
    print("不及格...")

# ===== 代码块 19 =====

score = 85
# 创建一个变量 score，值是 85

if score >= 90:
# if = 如果
# >= = 大于等于
# score >= 90 = 如果分数大于等于 90
# : = 冒号（不能少！表示后面是要执行的代码）

    print("优秀！")
    # 这行前面有 4 个空格（叫缩进）
    # 表示这行属于 if 语句
    # 只有 score >= 90 时才会执行
    
elif score >= 80:
# elif = 否则如果（else if 的缩写）
# 如果上面的条件不满足，就检查这个条件

    print("良好！")
    # 同样，前面要有缩进
    
elif score >= 60:
    print("及格！")
    
else:
# else = 否则
# 以上条件都不满足时，执行这里的代码

    print("不及格...")

# ===== 代码块 20 =====

# 天气判断
weather = "晴天"

if weather == "晴天":
    print("去公园玩！")
elif weather == "雨天":
    print("在家看书")
elif weather == "雪天":
    print("堆雪人！")
else:
    print("随便逛逛")

# 年龄判断
age = 20

if age < 18:
    print("未成年")
elif age >= 18 and age < 60:
    print("成年人")
else:
    print("老年人")

# ===== 代码块 21 =====

# 打印 1 到 5
for i in range(1, 6):
    print(i)

# 输出：
# 1
# 2
# 3
# 4
# 5

# ===== 代码块 22 =====

for i in range(1, 6):
# for = 对于
# i = 一个变量，每次循环会变
# in = 在...里面
# range(1, 6) = 范围从 1 到 6（不包括 6）
# : = 冒号（不能少）
# 整句话：对于 1 到 5 的每一个数字 i，执行下面的代码

    print(i)
    # 缩进
    # 打印 i 的值

# ===== 代码块 23 =====

range(5)      # 生成 0, 1, 2, 3, 4（5 个数，从 0 开始）
range(1, 6)   # 生成 1, 2, 3, 4, 5（5 个数，从 1 开始）
range(1, 10, 2)  # 生成 1, 3, 5, 7, 9（每隔 2 个数）

# ===== 代码块 24 =====

fruits = ["苹果", "香蕉", "橙子", "葡萄"]

for fruit in fruits:
    print("我喜欢吃", fruit)

# 输出：
# 我喜欢吃 苹果
# 我喜欢吃 香蕉
# 我喜欢吃 橙子
# 我喜欢吃 葡萄

# ===== 代码块 25 =====

# 数数
print("从 1 数到 10：")
for i in range(1, 11):
    print(i)

# 遍历爱好
hobbies = ["打游戏", "看电影", "读书"]
print("\n我的爱好：")
for hobby in hobbies:
    print("-", hobby)

# 计算 1 加到 100
total = 0
for i in range(1, 101):
    total = total + i
    
print("\n1 加到 100 的和是：", total)

# ===== 代码块 26 =====

# 定义一个函数
def greet(name):
    """打招呼"""
    return "你好，" + name + "！"

# 使用函数
message = greet("张三")
print(message)   # 输出：你好，张三！

message = greet("李四")
print(message)   # 输出：你好，李四！

# ===== 代码块 27 =====

def greet(name):
# def = define（定义）的缩写
# greet = 函数的名字（你可以自己起）
# (name) = 参数（输入的东西）
# : = 冒号（不能少）
# 整句话：定义一个叫 greet 的函数，需要一个参数 name

    """打招呼"""
    # 三个引号 = 文档字符串（说明这个函数是干嘛的）
    # 不是必须的，但建议写上
    
    return "你好，" + name + "！"
    # return = 返回
    # 后面的东西会作为结果返回
    # + = 连接字符串

# ===== 代码块 28 =====

# 定义加法函数
def add(a, b):
    """计算两个数的和"""
    result = a + b
    return result

# 使用
sum1 = add(3, 5)
print(sum1)   # 输出：8

sum2 = add(10, 20)
print(sum2)   # 输出：30

# ===== 代码块 29 =====

# 创建一个乘法函数
def multiply(x, y):
    """计算两个数的乘积"""
    return x * y

# 测试
result1 = multiply(3, 4)
print("3 × 4 =", result1)

result2 = multiply(5, 6)
print("5 × 6 =", result2)

# 创建一个判断成年的函数
def is_adult(age):
    """判断是否成年"""
    if age >= 18:
        return True
    else:
        return False

# 测试
print("18 岁成年吗？", is_adult(18))
print("16 岁成年吗？", is_adult(16))

# ===== 代码块 30 =====

# 创建一个字典，包含你的信息
# 然后用 print 显示出来

# ===== 代码块 31 =====

# 写一个函数，可以做加减乘除
# 测试一下

# ===== 代码块 32 =====

# 电脑随机想一个 1-100 的数字
# 你有 10 次机会猜
# 每次告诉你大了还是小了