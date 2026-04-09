# 🎓 AI 入门 30 天挑战 - Day 1 真正零基础版

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **完全看不懂编程？没关系！**  
> **这个教程会像教小学生一样，一步一步带你！**  
> **每个概念都解释，每行代码都说明白！**

---

## 📖 先说说：什么是 AI？什么是编程？

### 故事时间 📚

想象你在教一个**外星机器人**做菜：

```
你想让机器人帮你炒鸡蛋：

❌ 错误说法："炒个鸡蛋"
机器人：？？？（听不懂）

✅ 正确说法：
1. 走到冰箱前
2. 打开冰箱门
3. 拿出一个鸡蛋
4. 关上冰箱门
5. 走到灶台前
6. 拿起锅
7. 开火
8. 倒油
9. 打鸡蛋
10. 翻炒
11. 关火
12. 装盘

机器人：好的！（执行）
```

**编程就是这样！**
- 用电脑能懂的语言（比如 Python）
- 一步一步告诉电脑要做什么
- 电脑就会乖乖执行

**AI（人工智能）是什么？**
```
普通程序 = 你告诉电脑每一步怎么做
AI 程序 = 电脑自己学习怎么做

例子：
普通程序 → 你写规则：如果是猫的照片，就输出"猫"
AI 程序 → 给电脑看 1000 张猫的照片，它自己学会认猫
```

---

## 🐍 什么是 Python？

**Python = 一种编程语言**

就像人类有：
- 中文（中国人用）
- 英文（英国人用）
- 日文（日本人用）

电脑也有自己的语言：
- Python（科学家、AI 专家用）
- Java（企业开发用）
- C++（游戏、系统开发用）

**Python 的特点：**
- ✅ 简单易懂（最接近人类语言）
- ✅ 功能强大（什么都能做）
- ✅ 很流行（很多人用）

**特别适合：**
- 新手入门（就是你！）
- 数据分析（分析销售数据、用户行为）
- 人工智能（教电脑认图片、理解语言）
- 网站开发（做淘宝、京东这样的网站）

---

## 🚀 第 1 步：安装 Python（超级详细版）

### 方法 1：安装 Anaconda（强烈推荐！最简单）

**Anaconda 是什么？**

想象你要学做饭：
- 纯 Python = 只给你食材（还要自己去买锅碗瓢盆）
- Anaconda = 厨房大礼包（食材 + 锅碗瓢盆 + 菜谱，全都有！）

**所以我们推荐 Anaconda，什么都包含了！**

#### 详细安装步骤：

**第 1 步：下载安装包**

```
1. 打开你的浏览器（Chrome、Edge 都可以）
2. 在地址栏输入（或复制粘贴）：
   https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
   
   这是清华大学的镜像站（下载快！）
   
3. 按回车键，会看到一个文件列表
4. 找最新的 Windows 版本
   文件名类似：Anaconda3-2024.02-1-Windows-x86_64.exe
   
5. 点击这个文件，开始下载
6. 等下载完成（文件比较大，可能需要几分钟）
```

**第 2 步：安装**

```
1. 找到下载的文件（通常在"下载"文件夹）
2. 双击这个文件
3. 出现安装向导，点 Next
4. 看到许可协议，选 I Agree（同意）
5. 选择安装对象，选 Just Me（只有你用）
6. 选择安装路径：
   - 默认是 C 盘
   - 如果 C 盘空间不够，可以改到 D 盘
   - 建议保持默认
7. ⚠️ 重要！看到这两个选项都要勾选：
   ☑ Add Anaconda3 to the PATH environment variable
   ☑ Register Anaconda3 as the system Python 3.x
   
   不勾选的话，后面可能用不了！
   
8. 点 Install（安装）
9. 等待 20-30 分钟（时间比较长，可以喝杯水休息一下）
10. 看到 Finish，安装完成！
```

**第 3 步：验证安装成功**

```
1. 按键盘上的 Win 键（Windows 图标那个键）
2. 输入：cmd
3. 会出现一个叫"命令提示符"的程序，点击打开
4. 在黑黑的窗口里输入：
   python --version
   
5. 按回车
6. 如果显示 Python 3.x.x 就成功了！

例如：
C:\Users\YourName>python --version
Python 3.11.5

这就对了！✅
```

---

## 🎯 第 2 步：第一个 Python 程序（超简单）

### 方式 1：用最简单的记事本

**第 1 步：打开记事本**

```
1. 按 Win 键
2. 输入：记事本
3. 点击打开
```

**第 2 步：写代码**

在记事本里输入以下内容（注意大小写和标点符号）：

```python
print("你好，世界！")
print("这是我的第一个 Python 程序！")
print(1 + 1)
print("我今年", 25, "岁")
```

**解释每一行：**

```python
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
```

**第 3 步：保存文件**

```
1. 点击记事本菜单的 文件 → 另存为
2. 选择一个好找的位置（比如桌面）
3. 文件名：test.py
   ⚠️ 注意：一定是 .py 结尾（这是 Python 文件的标志）
4. 保存类型：所有文件（*.*）
   ⚠️ 不要选文本文档（*.txt）
5. 编码：UTF-8
   ⚠️ 这个重要，不然中文会乱码
6. 点 保存
```

**第 4 步：运行程序**

```
方法 1：右键运行
1. 找到刚才保存的 test.py 文件
2. 右键点击它
3. 选择 打开方式 → Python
4. 会看到一个黑窗口闪过（跑得很快）

方法 2：命令行运行（推荐）
1. 按 Win + R
2. 输入：cmd
3. 按回车
4. 在黑窗口里输入：
   cd Desktop
   （如果你的文件在桌面）
   
5. 然后输入：
   python test.py
   
6. 按回车
```

**你会看到输出：**

```
你好，世界！
这是我的第一个 Python 程序！
2
我今年 25 岁
```

**恭喜你！你已经写出了第一个 Python 程序！** 🎉

---

## 💻 第 3 步：用 Jupyter Notebook（推荐工具）

### 什么是 Jupyter Notebook？

**Jupyter Notebook = 一个很好用的编程工具**

就像一个笔记本：
- 可以写代码
- 可以立即看到结果
- 可以写笔记
- 可以贴图

**比记事本好用 100 倍！**

### 启动 Jupyter Notebook

**如果你装了 Anaconda，已经有 Jupyter 了！**

**方法 A：图形界面启动（推荐新手）**

```
1. 按 Win 键
2. 找 "Anaconda Navigator"（有个绿色图标）
3. 点击打开（可能要等几秒）
4. 打开后看到很多应用的界面
5. 找 "Jupyter Notebook"（橙色图标）
6. 点右边的 "Launch" 按钮
7. 会自动打开你的浏览器
8. 看到文件列表就成功了！
```

**方法 B：命令行启动（更快）**

```
1. 按 Win + R
2. 输入：cmd
3. 按回车
4. 在黑窗口里输入：
   jupyter notebook
   
5. 按回车
6. 等几秒
7. 自动打开浏览器
8. 地址通常是：http://localhost:8888
```

### 创建第一个笔记本

**在 Jupyter 里写代码：**

```
1. 在 Jupyter 页面，点击右上角的 "New" 按钮
2. 选择 "Python 3"
3. 会打开一个新标签页
4. 看到一个空白格子（叫 Cell）
5. 还有一排工具栏
```

**试试写代码：**

在第 1 个格子里输入：

```python
# 这是我的第一行代码
print("Hello, AI!")
print("我正在学习 Python")
print(100 + 200)
```

**运行代码：**

```
方法 1：按 Shift + Enter（推荐）
方法 2：点击工具栏的 ▶️ Run 按钮
```

**你会看到：**

```
Hello, AI!
我正在学习 Python
300
```

**下面还会出现一个新的格子，可以继续写！**

---

## 📦 第 4 步：学习 Python 基础（像搭积木一样）

### 1. 变量 - 数据的"盒子"

**什么是变量？**

想象你有很多收纳盒：
- 盒子上贴标签（变量名）
- 盒子里装东西（数据）

**例子：**

```python
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
```

**详细解释：**

```python
age = 25
# age = 变量的名字（盒子的标签）
# = = 赋值符号（把右边的东西放进左边的盒子）
# 25 = 数据（盒子里的东西）
# 整句话：创建一个叫 age 的盒子，里面放数字 25
```

**在 Jupyter 里试试：**

打开 Jupyter，新建一个笔记本，在格子里输入：

```python
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
```

按 Shift + Enter 运行，你会看到：

```
姓名： 张三
年龄： 25 岁
身高： 1.75 米
体重： 65.5 公斤
是学生吗？ True
```

**True = 是的（对）**  
**False = 不是（错）**

---

### 2. 数据类型 - 盒子里装的东西

**Python 里有几种常见的数据：**

#### （1）整数（int）- 没有小数的数字

```python
age = 25          # 年龄
count = 100       # 数量
price = 50        # 价格（元）
negative = -10    # 负数

print(age)        # 输出：25
print(type(age))  # 看看这是什么类型
                  # 输出：<class 'int'>
```

**int = integer（整数）的缩写**

#### （2）小数（float）- 有小数点的数字

```python
height = 1.75     # 身高（米）
weight = 65.5     # 体重（公斤）
score = 95.5      # 分数
pi = 3.14159      # 圆周率

print(height)     # 输出：1.75
print(type(height))  # 输出：<class 'float'>
```

**float = floating point（浮点数）的缩写**

#### （3）文字（string）- 用引号括起来的文字

```python
name = "张三"           # 名字
city = '北京'           # 城市
message = "你好世界"    # 消息
empty = ""              # 空字符串

print(name)         # 输出：张三
print(type(name))   # 输出：<class 'str'>
```

**str = string（字符串）的缩写**

**注意：**
- 可以用双引号 `" "`
- 也可以用单引号 `' '`
- 但要成对出现

#### （4）对错（boolean）- 只有两个值

```python
is_student = True     # 是学生
is_raining = False    # 没下雨
has_money = True      # 有钱
is_adult = False      # 不是成年人

print(is_student)     # 输出：True
print(type(is_student))  # 输出：<class 'bool'>
```

**bool = boolean（布尔）的缩写**  
**True = 真（对、是）**  
**False = 假（错、否）**

**在 Jupyter 里练习：**

```python
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
```

---

### 3. 列表（list）- 一排盒子

**什么是列表？**

想象一排连续的收纳盒：
```
[盒子 1, 盒子 2, 盒子 3, ...]
```

每个盒子都有编号（从 0 开始）：
- 第 1 个盒子 → 编号 0
- 第 2 个盒子 → 编号 1
- 第 3 个盒子 → 编号 2
- ...

**例子：**

```python
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
```

**详细解释：**

```python
hobbies = ["打游戏", "看电影", "读书", "运动"]
# hobbies = 列表的名字
# [] = 列表的标志
# [] 里的内容 = 列表的元素
# 每个元素用引号括起来（因为是文字）
# 元素之间用逗号隔开
```

**在 Jupyter 里练习：**

```python
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
```

**len() = length（长度）的缩写**

---

### 4. 字典（dict）- 带标签的盒子

**什么是字典？**

不是用数字编号，而是用标签取东西：
```
{
  "姓名": "张三",
  "年龄": 25,
  "城市": "北京"
}
```

**例子：**

```python
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
```

**详细解释：**

```python
person = {
    "name": "张三",
    # "name" = 键（key），就是标签
    # "张三" = 值（value），就是内容
    # : = 冒号，连接键和值
}
# {} = 字典的标志
```

**在 Jupyter 里练习：**

```python
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
```

---

## 🎮 第 5 步：if 判断 - 做选择

### 什么时候需要做选择？

生活中经常要做选择：
```
如果明天下雨 → 带伞
否则 → 不带伞

如果考试及格 → 开心
否则 → 难过

如果有钱 → 买 iPhone
否则 → 买小米
```

### 代码怎么写？

```python
score = 85  # 考试分数

if score >= 90:
    print("优秀！")
elif score >= 80:
    print("良好！")
elif score >= 60:
    print("及格！")
else:
    print("不及格...")
```

**逐行解释：**

```python
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
```

**注意：**
- `:` = 冒号（不能少！）
- 缩进 = 前面的空格（必须对齐！）
- Python 用缩进来判断哪些代码属于一起的

**在 Jupyter 里练习：**

```python
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
```

**== = 等于（判断是否相等）**  
**and = 并且**

---

## 🔄 第 6 步：for 循环 - 重复做事

### 什么时候需要循环？

```
要把 100 个苹果装箱子

不用循环：要写 100 次代码
  print("装第 1 个苹果")
  print("装第 2 个苹果")
  ...
  print("装第 100 个苹果")
  
用循环：只要写 1 次
  for i in range(100):
      print("装第", i+1, "个苹果")
```

### 代码怎么写？

```python
# 打印 1 到 5
for i in range(1, 6):
    print(i)

# 输出：
# 1
# 2
# 3
# 4
# 5
```

**详细解释：**

```python
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
```

**range() 函数：**

```python
range(5)      # 生成 0, 1, 2, 3, 4（5 个数，从 0 开始）
range(1, 6)   # 生成 1, 2, 3, 4, 5（5 个数，从 1 开始）
range(1, 10, 2)  # 生成 1, 3, 5, 7, 9（每隔 2 个数）
```

**遍历列表：**

```python
fruits = ["苹果", "香蕉", "橙子", "葡萄"]

for fruit in fruits:
    print("我喜欢吃", fruit)

# 输出：
# 我喜欢吃 苹果
# 我喜欢吃 香蕉
# 我喜欢吃 橙子
# 我喜欢吃 葡萄
```

**在 Jupyter 里练习：**

```python
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
```

---

## 🛠️ 第 7 步：函数 - 打包好的工具

### 什么是函数？

就像一个电器：
- 插上电（输入参数）
- 工作（处理）
- 出结果（返回值）

或者像一个机器：
```
绞肉机：
  输入：肉块
  处理：绞碎
  输出：肉末
  
函数：
  输入：参数
  处理：代码
  输出：返回值
```

### 例子：

```python
# 定义一个函数
def greet(name):
    """打招呼"""
    return "你好，" + name + "！"

# 使用函数
message = greet("张三")
print(message)   # 输出：你好，张三！

message = greet("李四")
print(message)   # 输出：你好，李四！
```

**详细解释：**

```python
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
```

**再举个例子：**

```python
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
```

**在 Jupyter 里练习：**

```python
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
```

---

## 🎉 今日总结

### ✅ 你今天学到了：

**1. 什么是编程**
- 用电脑懂的语言告诉它做什么

**2. Python 基础**
- 变量 = 装数据的盒子
- 数据类型 = 整数、小数、文字、对错
- 列表 = 一排盒子
- 字典 = 带标签的盒子

**3. 控制结构**
- if 判断 = 做选择
- for 循环 = 重复做事

**4. 函数**
- 打包好的工具
- 输入 → 处理 → 输出

**5. 安装了 Python 环境**
- Anaconda（推荐）
- Jupyter Notebook（好用的工具）

**6. 写了第一个程序**
- print() 函数
- 简单的计算

---

## 🎯 下一步做什么？

### 明天学习：
- NumPy 基础（处理数字的神器）
- 数组操作
- 为机器学习做准备

### 今天可以做的练习：

**练习 1：自我介绍**
```python
# 创建一个字典，包含你的信息
# 然后用 print 显示出来
```

**练习 2：计算器**
```python
# 写一个函数，可以做加减乘除
# 测试一下
```

**练习 3：猜数字游戏**
```python
# 电脑随机想一个 1-100 的数字
# 你有 10 次机会猜
# 每次告诉你大了还是小了
```

---

## 💡 学习建议

### ✅ 应该做的：

1. **每天都敲代码** - 不要只看，要动手！
2. **每个例子都运行** - 看结果，加深理解
3. **不懂就问** - Google、问 AI、问朋友
4. **允许自己犯错** - 这很正常
5. **慢慢来** - 理解比速度重要

### ❌ 不应该做的：

1. **不要死记硬背** - 理解最重要
2. **不要追求完美** - 先完成再说
3. **不要跟别人比** - 按自己的节奏
4. **不要轻易放弃** - 每个人都这样过来的

---

## 🆘 遇到问题怎么办？

### 常见问题：

**Q1: 代码报错怎么办？**
```
A: 
1. 仔细看错误信息（通常会告诉你哪里错了）
2. 复制到 Google 搜索
3. 检查拼写错误（大小写、标点符号）
4. 问 AI 助手
```

**Q2: 看不懂怎么办？**
```
A:
1. 多读几遍
2. 换个教程看看
3. 找人问
4. 先跳过，后面可能会明白
```

**Q3: 学得太慢怎么办？**
```
A:
1. 这很正常！
2. 每个人学习速度不同
3. 重要的是坚持
4. 完成比完美重要
```

---

## 🌟 最后的话

**恭喜你完成了 Python 入门！**

你可能觉得：
- 有些地方还是不懂 → 正常！
- 学得有点累 → 休息下再来！
- 不知道有没有用 → 继续学就知道！

**但请记住：**
- 每个程序员都是从零开始的
- 你已经迈出了最重要的一步
- 坚持下去，你一定会惊讶于自己的成长

**加油！我相信你一定可以的！** 💪🎉

---

## 📞 打卡模板

```
日期：___________
学习时长：_______ 小时
掌握程度：⭐⭐⭐⭐⭐

今天学会了：


遇到的问题：


明天的目标：


```

**明天见！继续加油！** ✨

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

- [→ Day02](../Day02/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*
