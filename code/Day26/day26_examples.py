"""
Day26 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day26_examples.py

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
print("Day26 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

"""
监督学习 (Supervised Learning):

老师教学生:
题目："1+1=?"
答案："2"

学生：记住就好了

特点:
✓ 有标准答案
✓ 数据是现成的
✗ 不会主动探索

应用:
- 图像分类
- 情感分析
- 语音识别


无监督学习 (Unsupervised Learning):

给学生一堆数据:
"你们自己找规律吧"

学生：？？？

特点:
✓ 不需要标注数据
✓ 自动发现结构
✗ 没有明确目标

应用:
- 聚类分析
- 降维
- 异常检测


强化学习 (Reinforcement Learning):

让学生自己做题:
做对了 → 奖励
做错了 → 惩罚

学生：那我试试这样...哎呀不对，换一种...对了!

特点:
✓ 从试错中学习
✓ 长期回报最大化
✓ 主动探索环境

应用:
- 游戏 AI(AlphaGo、Dota2)
- 机器人控制
- 自动驾驶
- 推荐系统
"""

# ===== 代码块 2 =====

"""
强化学习的关键术语:

1. Agent(智能体)
   - 学习者和决策者
   - 比如：游戏里的角色、机器人

2. Environment(环境)
   - Agent 之外的所有东西
   - 比如：游戏世界、现实世界

3. State(状态)
   - 环境的当前情况
   - 比如：游戏画面、机器人位置

4. Action(动作)
   - Agent 能做的事情
   - 比如：上下左右移动、跳跃

5. Reward(奖励)
   - 环境的反馈
   - 比如：得分、扣血、获胜

6. Policy(策略)
   - Agent 的行为规则
   - 比如："看到敌人就跑"

7. Value(价值)
   - 长期回报的期望
   - 比如：这个位置未来能得多少分
"""

# ===== 代码块 3 =====

"""
强化学习的核心矛盾:

探索 (Exploration):
- 尝试新的动作
- 可能会发现更好的策略
- 但可能犯错、扣分

利用 (Exploitation):
- 用已知的最好策略
- 稳定得分
- 但可能错过更好的方法

例子:
你去餐厅吃饭

探索:
- 点没吃过的菜
- 可能很难吃 (踩雷)
- 也可能发现新大陆 (惊喜)

利用:
- 点常吃的宫保鸡丁
- 不会难吃 (安全)
- 但也吃不到新口味

怎么办？
ε-greedy 策略:
- 大部分时候 (90%) 用最好的
- 偶尔 (10%) 尝试新的
- 平衡探索和利用
"""

# ===== 代码块 4 =====

"""
Q-Learning 是最经典的强化学习算法

Q 是什么？
Q(s, a) = 在状态 s 下做动作 a 的价值

说人话:
Q(看到敌人，逃跑) = 80 分
Q(看到敌人，攻击) = 20 分
→ 所以应该逃跑!

怎么学习 Q 值？
用 Bellman 方程 (不用怕，很简单):

Q(s,a) ← Q(s,a) + α * [r + γ*max(Q(s',a')) - Q(s,a)]

翻译成人话:
新的 Q 值 = 旧的 Q 值 + 学习率 * [奖励 + 折扣因子*未来的最大 Q 值 - 旧的 Q 值]

就像考试后订正:
你做对了题 (得到奖励)
老师告诉你这题多重要 (折扣因子)
你记住了这个知识点 (更新 Q 值)
"""

# ===== 代码块 5 =====

"""
Q-Table 就是一个表格，记录所有状态 - 动作对的 Q 值

例子 (简化版迷宫):

状态：位置 (1,1), (1,2), ..., (3,3)
动作：上、下、左、右

Q-Table:
┌─────────┬──────┬──────┬──────┬──────┐
│ 状态    │ 上   │ 下   │ 左   │ 右   │
├─────────┼──────┼──────┼──────┼──────┤
│ (1,1)   │ 0.5  │ 0.2  │ 0.1  │ 0.8  │ ← 应该往右
│ (1,2)   │ 0.9  │ 0.3  │ 0.4  │ 0.2  │ ← 应该往上
│ ...     │ ...  │ ...  │ ...  │ ...  │
└─────────┴──────┴──────┴──────┴──────┘

问题:
如果状态太多怎么办？
比如游戏画面：210×160×3 种颜色组合
→ 天文数字！表格放不下!

解决:
用神经网络近似 Q 值!
→ 这就是 Deep Q-Learning
"""

# ===== 代码块 6 =====

import numpy as np

class QLearningAgent:
    """Q-Learning 智能体"""
    
    def __init__(self, n_states, n_actions, 
                 learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        参数:
        n_states: 状态数量
        n_actions: 动作数量
        learning_rate(α): 学习率 (0.1 表示每次更新 10%)
        discount_factor(γ): 折扣因子 (0.9 表示未来奖励打 9 折)
        epsilon: 探索概率 (0.1 表示 10% 的概率随机探索)
        """
        
        # 初始化 Q 表 (全为 0)
        self.q_table = np.zeros((n_states, n_actions))
        
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.n_actions = n_actions
    
    def choose_action(self, state):
        """
        选择动作 (ε-greedy 策略)
        
        大部分时候选最好的，偶尔随机
        """
        
        if np.random.random() < self.epsilon:
            # 探索：随机选一个动作
            action = np.random.randint(self.n_actions)
        else:
            # 利用：选 Q 值最大的动作
            action = np.argmax(self.q_table[state, :])
        
        return action
    
    def update(self, state, action, reward, next_state):
        """
        更新 Q 值
        
        Bellman 方程的实现
        """
        
        # 当前 Q 值
        current_q = self.q_table[state, action]
        
        # 未来的最大 Q 值
        max_future_q = np.max(self.q_table[next_state, :])
        
        # 计算新的 Q 值
        new_q = current_q + self.learning_rate * \
                (reward + self.discount_factor * max_future_q - current_q)
        
        # 更新 Q 表
        self.q_table[state, action] = new_q
    
    def get_best_action(self, state):
        """获取最佳动作 (不探索，只利用)"""
        return np.argmax(self.q_table[state, :])

# 测试 Q-Learning
print("=" * 60)
print("Q-Learning 演示 - 简单迷宫")
print("=" * 60)

"""
创建一个简单的迷宫环境:

S = 起点 (0)
G = 终点 (5)
_ = 普通格子
X = 陷阱

S _ X _ _ G
0 1 2 3 4 5

规则:
- 走到终点：+100 分
- 走到陷阱：-100 分
- 每走一步：-1 分 (鼓励快点到终点)
"""

# 环境配置
n_states = 6
n_actions = 2  # 左、右 (简化版，只能左右移动)

# 定义环境
goal_state = 5
trap_state = 2

# 创建智能体
agent = QLearningAgent(n_states, n_actions)

print("\n开始训练...")
print("训练 1000 次，看看能不能学会避开陷阱\n")

# 训练
for episode in range(1000):
    state = 0  # 从起点开始
    total_reward = 0
    
    while state != goal_state and state != trap_state:
        # 选择动作
        action = agent.choose_action(state)
        
        # 执行动作 (简化版：0=左，1=右)
        if action == 0:  # 左
            next_state = max(0, state - 1)
        else:  # 右
            next_state = min(n_states - 1, state + 1)
        
        # 计算奖励
        if next_state == goal_state:
            reward = 100  # 到达终点
        elif next_state == trap_state:
            reward = -100  # 掉进陷阱
        else:
            reward = -1  # 每步扣 1 分
        
        # 更新 Q 值
        agent.update(state, action, reward, next_state)
        
        state = next_state
        total_reward += reward
    
    # 每 100 次打印一次进度
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode+1}: 总奖励 = {total_reward}")

print("\n训练完成!")
print("\n最终的 Q 表:")
print(agent.q_table)

print("\n学到的策略:")
for state in range(n_states):
    best_action = agent.get_best_action(state)
    action_name = "右" if best_action == 1 else "左"
    print(f"位置{state}: 最佳动作 = {action_name}")

print("\n结论:")
print("- 起点 (0): 应该往右走")
print("- 位置 1: 应该往右走 (虽然右边是陷阱，但更右边是终点)")
print("- 位置 2(陷阱): 已经挂了，无所谓")
print("- 位置 3: 应该往右走")
print("- 位置 4: 应该往右走 (快到终点了!)")

# ===== 代码块 7 =====

"""
Q-Learning 的问题:

Q-Table 太小了!

例子 (Atari 游戏):
- 屏幕：210×160 像素
- 每个像素：3 通道 (RGB)
- 每个通道：256 种颜色

状态数 = 210 × 160 × 256^3 ≈ 10^100000

什么概念？
- 比宇宙中的原子还多!
- 存不下!根本建不了 Q-Table!

怎么办？
用神经网络近似 Q 值!

输入：游戏画面
      ↓
卷积神经网络
      ↓
输出：每个动作的 Q 值

这样就不用存表格了!
"""

# ===== 代码块 8 =====

"""
DQN 的两个关键技术:

1. Experience Replay(经验回放)

问题:
- 连续的游戏帧是相关的
- 直接训练会导致"遗忘"

解决:
- 把经验存起来 (s, a, r, s')
- 随机采样 batch 来训练
- 打破相关性

就像学习:
- 不要只记最近的经验
- 要复习以前的错题
- 全面总结才能进步


2. Target Network(目标网络)

问题:
- Q 值的目标一直在变
- 训练不稳定

解决:
- 用两个网络
- 一个主网络 (经常更新)
- 一个目标网络 (偶尔更新)
- 让训练更稳定

就像考试:
- 学习目标不能天天变
- 定好目标，坚持一段时间
- 再根据情况调整
"""

# ===== 代码块 9 =====

import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    Deep Q-Network
    
    输入：游戏画面 (4 帧堆叠，84×84)
    输出：每个动作的 Q 值
    """
    
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        
        # 卷积层提取特征
        # 输入：(batch_size, 4, 84, 84)
        # 4 帧堆叠，让网络看到运动信息
        
        self.conv1 = nn.Conv2d(
            in_channels=4,      # 4 帧灰度图
            out_channels=32,    # 32 个卷积核
            kernel_size=8,      # 8×8 卷积核
            stride=4            # 步长 4 (缩小 4 倍)
        )
        # 输出：(batch_size, 32, 20, 20)
        
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=2
        )
        # 输出：(batch_size, 64, 8, 8)
        
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        # 输出：(batch_size, 64, 6, 6)
        
        # 全连接层
        # 先计算卷积输出的维度
        conv_output_size = 64 * 6 * 6
        
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
        self.n_actions = n_actions
    
    def forward(self, x):
        """前向传播"""
        
        # 卷积层 + ReLU 激活
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values

# 测试 DQN
print("\n" + "=" * 60)
print("DQN 网络测试")
print("=" * 60)

n_actions = 4  # 假设 4 个动作 (上下左右)
model = DQN(n_actions)

# 模拟输入 (batch_size=2, 4 帧，84×84)
dummy_input = torch.randn(2, 4, 84, 84)
output = model(dummy_input)

print(f"输入形状：{dummy_input.shape}")
print(f"输出形状：{output.shape}")
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")
print(f"\n✓ DQN 网络创建成功!")

print("""
DQN 网络结构详解:

输入层:
- 4 帧 84×84 灰度图
- 为什么 4 帧？看到运动信息
  (就像动画，单张看不出运动)

卷积层:
- conv1: 32 个卷积核，8×8，stride=4
  → 提取低级特征 (边缘、角点)
- conv2: 64 个卷积核，4×4，stride=2
  → 提取中级特征 (纹理、图案)
- conv3: 64 个卷积核，3×3，stride=1
  → 提取高级特征 (物体、形状)

全连接层:
- fc1: 512 个神经元
  → 整合所有特征
- fc2: n_actions 个输出
  → 每个动作的 Q 值

输出:
- [Q(上), Q(下), Q(左), Q(右)]
- 选 Q 值最大的动作
""")

# ===== 代码块 10 =====

"""
DQN 训练伪代码:

初始化:
- 主网络 Q
- 目标网络 Q_target (参数复制自 Q)
- 经验回放池 ReplayBuffer

对于每一个 episode:
    state = 环境.reset()
    
    对于每一步:
        # 1. 选择动作 (ε-greedy)
        if 随机数 < ε:
            action = 随机动作 (探索)
        else:
            action = argmax(Q(state, :)) (利用)
        
        # 2. 执行动作
        next_state, reward, done = env.step(action)
        
        # 3. 存储经验
        ReplayBuffer.add(state, action, reward, next_state, done)
        
        # 4. 训练网络
        从 ReplayBuffer 采样 batch
        
        # 计算目标 Q 值
        q_targets = reward + γ * max(Q_target(next_state, :))
        
        # 计算损失
        loss = MSE(Q(state, action), q_targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 5. 定期更新目标网络
        if step % target_update_freq == 0:
            Q_target.load_state_dict(Q.state_dict())
        
        state = next_state
        
        if done:
            break
"""