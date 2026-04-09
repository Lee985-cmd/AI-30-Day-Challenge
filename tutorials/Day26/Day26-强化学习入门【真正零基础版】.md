# 🎮 Day26: 强化学习入门 - 让 AI 学会决策【真正零基础版】

---

## 🌟 完整项目和代码

本教程是 **AI 入门 30 天挑战** 系列的一部分！

- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **欢迎 Star 支持！**

---



> **AlphaGo、游戏 AI 的核心技术！从零理解强化学习的奥秘!**  
> **本教程：完整代码 + 详细讲解 + Flappy Bird AI 实战**

---

## 📚 目录

1. [强化学习是什么？](#强化学习是什么)
2. [从试错到智能决策](#从试错到智能决策)
3. [Q-Learning 基础](#q-learning 基础)
4. [Deep Q-Network(DQN)](#deep-q-networkdqn)
5. [实战：Flappy Bird AI](#实战 flappy-bird-ai)
6. [常见问题](#常见问题)

---

## 🤔 强化学习是什么？

### 说人话版本

想象一下这个场景:

```
训练一只小狗:

小狗做对了 (比如坐下):
→ 给零食奖励 🦴
→ 小狗明白了："哦，这样做有好吃的!"

小狗做错了 (比如随地大小便):
→ 批评惩罚 ❌
→ 小狗明白了："这样做会被骂"

经过多次训练:
小狗学会了各种技能!
```

**这就是强化学习的核心思想!**

- **Agent(智能体)** = 小狗
- **Environment(环境)** = 你家
- **Action(动作)** = 坐下、打滚等
- **Reward(奖励)** = 零食或批评

### 强化学习 vs 其他机器学习

```python
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
```

### 强化学习能做什么？

**真实应用场景:**

1. **游戏 AI**
   - AlphaGo(围棋)
   - AlphaStar(星际争霸 2)
   - OpenAI Five(Dota2)
   - 游戏 bots

2. **机器人控制**
   - 学会走路、跑步
   - 抓取物体
   - 复杂动作

3. **自动驾驶**
   - 决策何时变道
   - 停车策略
   - 节能驾驶

4. **推荐系统**
   - 抖音推荐算法
   - 淘宝商品推荐
   - Netflix 影片推荐

5. **金融交易**
   - 股票买卖决策
   - 投资组合优化
   - 风险管理

6. **资源调度**
   - 数据中心冷却
   - 电网调度
   - 交通信号控制

---

## 🎯 从试错到智能决策

### 核心概念详解

```python
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
```

### 强化学习的流程

```
第 1 步：观察环境
Agent: "我现在在哪？周围有什么？"
      ↓
第 2 步：选择动作
Agent: "我该往哪走？往上吧"
      ↓
第 3 步：执行动作
Agent: 执行"向上"的动作
      ↓
第 4 步：获得奖励
环境："你得了 10 分!"
      ↓
第 5 步：更新策略
Agent: "哦，往上走能得分，记住了!"
      ↓
重复 1-5，直到学得很好
```

### 探索 vs 利用

```python
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
```

---

## 🔍 Q-Learning 基础

### Q-Learning 是什么？

```python
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
```

### Q-Table(表格)

```python
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
```

### 从零实现 Q-Learning

让我们用一个简单的例子实现 Q-Learning:

```python
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
```

---

## 🧠 Deep Q-Network(DQN)

### 为什么需要 DQN?

```python
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
```

### DQN 的核心创新

```python
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
```

### DQN 架构详解

```python
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
```

### DQN 训练流程

```python
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
```

---

## 🎮 实战：Flappy Bird AI

让我们实现一个完整的 Flappy Bird AI!

```python
# ============================================================================
# 第一部分：导入库
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gymnasium as gym

print("=" * 60)
print("Flappy Bird AI - DQN 实战")
print("=" * 60)

# ============================================================================
# 第二部分：定义 DQN 网络
# ============================================================================

class DQN(nn.Module):
    """Flappy Bird 专用的 DQN 网络"""
    
    def __init__(self, n_actions=2):
        super(DQN, self).__init__()
        
        # Flappy Bird 状态比较简单
        # 我们用全连接层就够了
        
        # 输入：[鸟的高度，垂直速度，管子距离，管子高度差]
        input_dim = 4
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, n_actions)
        )
        
        self.n_actions = n_actions
    
    def forward(self, x):
        return self.net(x)

# ============================================================================
# 第三部分：经验回放池
# ============================================================================

class ReplayBuffer:
    """经验回放池"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样 batch"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# 第四部分：DQN 智能体
# ============================================================================

class DQNAgent:
    """DQN 智能体"""
    
    def __init__(self, state_dim=4, n_actions=2):
        # 超参数
        self.gamma = 0.99          # 折扣因子
        self.epsilon = 1.0         # 初始探索率
        self.epsilon_min = 0.01    # 最小探索率
        self.epsilon_decay = 0.995 # 探索率衰减
        self.learning_rate = 0.001 # 学习率
        self.batch_size = 32       # batch 大小
        self.target_update = 10    # 目标网络更新频率
        
        # 创建网络
        self.model = DQN(n_actions)
        self.target_model = DQN(n_actions)
        
        # 复制参数到目标网络
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 损失函数
        self.criterion = nn.SmoothL1Loss()
        
        # 经验回放池
        self.memory = ReplayBuffer(capacity=5000)
    
    def choose_action(self, state):
        """选择动作 (ε-greedy)"""
        
        if random.random() < self.epsilon:
            # 探索：随机动作
            return random.randint(0, 1)
        else:
            # 利用：选 Q 值最大的
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_tensor)
                return torch.argmax(q_values).item()
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """训练网络"""
        
        # 经验不够，不训练
        if len(self.memory) < self.batch_size:
            return
        
        # 采样 batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # 计算当前 Q 值
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标 Q 值
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = self.criterion(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_model(self):
        """更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ============================================================================
# 第五部分：简化的 Flappy Bird 环境
# ============================================================================

class SimpleFlappyBird:
    """
    简化版 Flappy Bird
    
    状态空间:
    - 鸟的高度 (0-1)
    - 垂直速度 (-1-1)
    - 到管子的水平距离 (0-1)
    - 管子高度差 (-1-1)
    
    动作空间:
    - 0: 什么都不做
    - 1: 向上飞
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置环境"""
        self.bird_y = 0.5           # 鸟的高度
        self.bird_velocity = 0      # 垂直速度
        self.pipe_x = 1.0           # 管子水平位置
        self.pipe_gap_y = 0.5       # 管子间隙高度
        self.pipe_passed = False    # 是否已通过管子
        
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        return np.array([
            self.bird_y,
            self.bird_velocity,
            self.pipe_x,
            self.pipe_gap_y - self.bird_y
        ], dtype=np.float32)
    
    def step(self, action):
        """执行动作"""
        
        # 物理模拟
        gravity = 0.5               # 重力
        jump_strength = -0.3        # 跳跃力度
        
        if action == 1:  # 跳跃
            self.bird_velocity = jump_strength
        
        # 应用重力
        self.bird_velocity += gravity * 0.02
        self.bird_y += self.bird_velocity
        
        # 管子移动
        self.pipe_x -= 0.03  # 管子向左移动
        
        # 检查碰撞
        reward = 0
        done = False
        
        # 撞到地面或天花板
        if self.bird_y <= 0 or self.bird_y >= 1:
            done = True
            reward = -100
        
        # 通过管子
        if self.pipe_x < 0 and not self.pipe_passed:
            reward = 10
            self.pipe_passed = True
        
        # 撞到管子
        if 0.45 < self.pipe_x < 0.55:  # 在管子位置
            gap_top = self.pipe_gap_y - 0.15  # 间隙大小
            gap_bottom = self.pipe_gap_y + 0.15
            
            if not (gap_top < self.bird_y < gap_bottom):
                done = True
                reward = -100
        
        # 重置管子
        if self.pipe_x < -0.1:
            self.pipe_x = 1.0
            self.pipe_gap_y = np.random.uniform(0.3, 0.7)
            self.pipe_passed = False
        
        return self._get_state(), reward, done

# ============================================================================
# 第六部分：开始训练!
# ============================================================================

print("\n创建环境和智能体...")

env = SimpleFlappyBird()
agent = DQNAgent(state_dim=4, n_actions=2)

print("✓ 环境创建成功")
print("✓ 智能体创建成功")

print("\n" + "=" * 60)
print("开始训练 DQN!")
print("=" * 60)

# 训练参数
n_episodes = 500       # 训练 500 局
max_steps = 1000       # 每局最多 1000 步
target_update_freq = 10  # 每 10 局更新一次目标网络

# 记录分数
scores = []
avg_scores = []

print(f"\n训练配置:")
print(f"  - 总局数：{n_episodes}")
print(f"  - 每局最大步数：{max_steps}")
print(f"  - Batch size: {agent.batch_size}")
print(f"  - 初始探索率：{agent.epsilon:.2f}")
print(f"  - 学习率：{agent.learning_rate}")
print()

best_score = -999

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    
    for step in range(max_steps):
        # 选择动作
        action = agent.choose_action(state)
        
        # 执行动作
        next_state, reward, done = env.step(action)
        
        # 存储经验
        agent.remember(state, action, reward, next_state, done)
        
        # 训练网络
        loss = agent.learn()
        
        # 更新状态
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    # 更新目标网络
    if episode % target_update_freq == 0:
        agent.update_target_model()
    
    # 衰减探索率
    agent.decay_epsilon()
    
    # 记录分数
    scores.append(total_reward)
    avg_score = np.mean(scores[-100:])  # 最近 100 局的平均分
    avg_scores.append(avg_score)
    
    # 更新最高分
    if total_reward > best_score:
        best_score = total_reward
    
    # 打印进度
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1}/{n_episodes}:")
        print(f"  得分：{total_reward}")
        print(f"  平均得分 (近 100 局): {avg_score:.2f}")
        print(f"  探索率：{agent.epsilon:.3f}")
        print(f"  最高分：{best_score}")
        print()

# ============================================================================
# 第七部分：测试训练成果
# ============================================================================

print("\n" + "=" * 60)
print("训练完成！测试最终效果...")
print("=" * 60)

# 关闭探索，完全利用学到的策略
agent.epsilon = 0

test_episodes = 10
test_scores = []

print(f"\n测试 {test_episodes} 局 (不使用探索):\n")

for i in range(test_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        total_reward += reward
    
    test_scores.append(total_reward)
    print(f"第{i+1}局：得分 = {total_reward}")

print(f"\n测试结果:")
print(f"  - 平均分：{np.mean(test_scores):.2f}")
print(f"  - 最高分：{max(test_scores)}")
print(f"  - 最低分：{min(test_scores)}")

# ============================================================================
# 第八部分：保存模型
# ============================================================================

print("\n" + "=" * 60)
print("保存训练好的模型...")
print("=" * 60)

# 保存模型参数
torch.save(agent.model.state_dict(), 'flappy_bird_dqn.pth')
print("✓ 模型已保存到 'flappy_bird_dqn.pth'")

# 保存训练历史
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump({'scores': scores, 'avg_scores': avg_scores}, f)
print("✓ 训练历史已保存到 'training_history.pkl'")

# ============================================================================
# 第九部分：可视化训练结果
# ============================================================================

print("\n生成训练曲线图...")

try:
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图 1: 原始分数
    ax1.plot(scores, alpha=0.5, label='单局分数', linewidth=1)
    ax1.plot(avg_scores, 'r-', label='平均分 (100 局)', linewidth=2)
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('训练过程 - 分数变化', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图 2: 探索率变化
    epsilons = [max(0.01, 1.0 * (0.995 ** i)) for i in range(n_episodes)]
    ax2.plot(epsilons, 'g-', linewidth=2)
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Epsilon', fontsize=12)
    ax2.set_title('探索率衰减曲线', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_training_results.png', dpi=150)
    print("✓ 训练曲线已保存为 'dqn_training_results.png'")
    plt.show()
    
except ImportError:
    print("提示：安装 matplotlib 可以看到训练曲线图")
    print("  pip install matplotlib")

# ============================================================================
# 第十部分：总结和下一步
# ============================================================================

print("\n" + "=" * 60)
print("🎉 恭喜你完成了 DQN 实战项目!")
print("=" * 60)

print("""
【今天学到了什么？】

✓ 强化学习的基本概念
  - Agent、Environment、State、Action、Reward
  - 探索 vs 利用的平衡

✓ Q-Learning 算法
  - Q-Table 的构建和更新
  - Bellman 方程的应用

✓ Deep Q-Network(DQN)
  - 用神经网络近似 Q 值
  - Experience Replay 经验回放
  - Target Network 目标网络

✓ 完整的项目实战
  - 从零搭建 Flappy Bird AI
  - 训练和调优
  - 模型保存和加载

【下一步可以学什么？】

1. 改进 DQN
   - Double DQN (解决高估问题)
   - Dueling DQN (分离状态和价值)
   - Prioritized Replay (优先经验回放)

2. 尝试其他算法
   - Policy Gradient(策略梯度)
   - A3C/A2C(Actor-Critic)
   - PPO(近端策略优化)

3. 挑战更复杂的游戏
   - Atari 游戏 (用原始像素输入)
   - MuJoCo(机器人控制)
   - StarCraft II(即时战略)

4. 实际应用
   - 推荐系统
   - 自动驾驶决策
   - 金融交易策略

【资源推荐】

在线课程:
- CS285(Berkeley 强化学习)
- RL Course (David Silver)

书籍:
- 《Reinforcement Learning: An Introduction》
- 《深度强化学习》(李宏毅)

工具库:
- Stable Baselines3(封装好的 RL 算法)
- Ray RLLib(分布式 RL)
- Gymnasium(强化学习环境)

记住:
强化学习很难，需要耐心调试!
不要轻易放弃，坚持下去就会有收获!

加油！未来的 AI 训练师! 🎮✨
""")

---

## 🔗 相关链接

### 🌐 项目资源
- 💻 **GitHub 仓库**: [https://github.com/Lee985-cmd/AI-30-Day-Challenge](https://github.com/Lee985-cmd/AI-30-Day-Challenge)
- 📖 **CSDN 专栏**: [https://blog.csdn.net/m0_67081842?type=blog](https://blog.csdn.net/m0_67081842?type=blog)
- ⭐ **如果觉得有帮助，请给 GitHub 仓库 Star 支持！**

### 📚 学习路径
- [← Day25](../Day25/README.md)
- [→ Day27](../Day27/README.md)

---

*本教程属于 [AI 入门 30 天挑战](https://github.com/Lee985-cmd/AI-30-Day-Challenge) 系列*
