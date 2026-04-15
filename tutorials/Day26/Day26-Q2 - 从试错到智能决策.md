# Day26-Q2 - 从试错到智能决策

## 🎯 智能决策的核心原理

### 试错学习的智慧

```python
"""
人类的学习过程:

小时候学走路:
- 走歪了 → 摔倒 → 痛 → 记住错误
- 走稳了 → 前进 → 成功 → 强化行为
- 经过无数次试错 → 学会走路

这就是强化学习的精髓:
通过试错 → 获得反馈 → 调整行为 → 优化策略
"""
```

### 智能决策的数学模型

#### 1. 马尔可夫决策过程 (MDP)

**核心思想：**
```
未来的状态只取决于当前状态和动作
与过去的历史无关

P(S_{t+1}|S_t, A_t) = P(S_{t+1}|S_0, A_0, S_1, A_1, ..., S_t, A_t)
```

**MDP 的五元组:**
1. **S**: 状态空间 (States)
2. **A**: 动作空间 (Actions)  
3. **P**: 状态转移概率 (Transition Probability)
4. **R**: 奖励函数 (Reward Function)
5. **γ**: 折扣因子 (Discount Factor)

**示例：机器人导航**
```
状态 S: 机器人在网格中的位置
动作 A: 上、下、左、右移动
转移 P: 移动后到达的新位置
奖励 R: 到达目标+10，撞墙-1，其他-0.1
折扣 γ: 0.9 (偏好近期奖励)
```

#### 2. 策略 (Policy)

**定义：** 从状态到动作的映射

**数学表示：**
```
π(a|s) = P[A_t=a | S_t=s]

确定性策略: a = π(s)
随机策略: π(a|s) = P(A=a|S=s)
```

**策略的类型：**
- **贪婪策略**: 总是选择当前最优动作
- **ε-贪婪策略**: 1-ε 概率选最优，ε 概率随机探索
- **软策略**: 根据概率分布选择动作

**示例：**
```python
import numpy as np

class EpsilonGreedyPolicy:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
    
    def get_action(self, q_values, legal_actions=None):
        """
        ε-贪婪策略
        
        参数:
        q_values: 每个动作的 Q 值
        legal_actions: 合法动作列表 (可选)
        
        返回:
        action: 选择的动作
        """
        
        if legal_actions is None:
            legal_actions = list(range(len(q_values)))
        
        # 随机探索
        if np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)
        
        # 选择最优动作
        legal_q_values = [q_values[a] for a in legal_actions]
        best_action_idx = np.argmax(legal_q_values)
        return legal_actions[best_action_idx]

# 使用示例
policy = EpsilonGreedyPolicy(epsilon=0.1)
q_values = [0.1, 0.8, 0.3, 0.6]  # 四个动作的 Q 值
action = policy.get_action(q_values)
print(f"选择的动作: {action}")
```

#### 3. 值函数 (Value Function)

**状态值函数 V(s):**
```
V_π(s) = E[G_t | S_t = s]  # 在策略 π 下，状态 s 的期望回报
```

**动作值函数 Q(s,a):**
```
Q_π(s,a) = E[G_t | S_t = s, A_t = a]  # 在策略 π 下，状态 s 执行动作 a 的期望回报
```

**Bellman 方程：**
```
V_π(s) = Σ π(a|s) Σ P(s'|s,a) [R(s,a,s') + γV_π(s')]
        a           s'

Q_π(s,a) = Σ P(s'|s,a) [R(s,a,s') + γ Σ π(a'|s')Q_π(s',a')]
            s'                           a'
```

### 从试错到优化的演进

#### 1. 朴素试错 (Naive Trial-and-Error)

**算法：**
```
1. 随机选择动作
2. 观察奖励
3. 调整行为
4. 重复
```

**问题：**
- 效率低下
- 容易遗忘
- 无法泛化

**示例：**
```python
class NaiveLearner:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.action_counts = np.zeros(n_actions)
        self.action_rewards = np.zeros(n_actions)
    
    def choose_action(self):
        # 随机选择动作
        return np.random.randint(self.n_actions)
    
    def update(self, action, reward):
        # 简单更新
        self.action_counts[action] += 1
        self.action_rewards[action] += reward

# 这种方法效率很低，没有利用历史信息
```

#### 2. 带记忆的试错 (Learning with Memory)

**改进：**
- 记录历史经验
- 计算平均回报
- 选择最佳动作

**算法：**
```python
class BanditLearner:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.action_counts = np.zeros(n_actions)
        self.action_values = np.zeros(n_actions)
    
    def choose_action(self):
        # ε-贪婪策略
        if np.random.random() < 0.1:  # 10% 探索
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.action_values)  # 90% 利用
    
    def update(self, action, reward):
        # 递增计数
        self.action_counts[action] += 1
        
        # 更新动作值 (增量更新)
        alpha = 1.0 / self.action_counts[action]  # 递减步长
        self.action_values[action] += alpha * (reward - self.action_values[action])

# 效果好于朴素试错，但仍有局限
```

#### 3. 智能决策 (Intelligent Decision Making)

**核心思想：**
- 状态感知
- 策略优化
- 长期规划

**Q-Learning 示例：**
```python
class QLearner:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
    
    def choose_action(self, state):
        # ε-贪婪策略
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.q_table[state]))
        else:
            return np.argmax(self.q_table[state])
    
    def learn(self, state, action, reward, next_state):
        # Q-Learning 更新
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

# 这是真正的强化学习算法！
```

## 🧠 智能决策的算法演进

### 1. 线性函数逼近 (Linear Function Approximation)

**问题：** 当状态空间很大时，Q-table 无法存储

**解决方案：** 使用特征函数逼近 Q 值

```python
class LinearQLearner:
    def __init__(self, n_features, n_actions, alpha=0.01, gamma=0.9):
        self.weights = np.random.normal(0, 0.01, (n_actions, n_features))
        self.alpha = alpha
        self.gamma = gamma
    
    def get_q_values(self, state_features):
        """计算状态的 Q 值"""
        return np.dot(self.weights, state_features)
    
    def choose_action(self, state_features):
        q_values = self.get_q_values(state_features)
        return np.argmax(q_values)
    
    def update(self, state_features, action, reward, next_state_features):
        """更新权重"""
        current_q = np.dot(self.weights[action], state_features)
        
        next_q_values = self.get_q_values(next_state_features)
        next_max_q = np.max(next_q_values)
        
        target = reward + self.gamma * next_max_q
        error = target - current_q
        
        # 更新权重
        self.weights[action] += self.alpha * error * state_features
```

### 2. 深度 Q 网络 (Deep Q-Network)

**突破：** 使用神经网络逼近 Q 函数

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNAgent:
    def __init__(self, input_dim, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(input_dim, n_actions).to(self.device)
        self.target_network = DQN(input_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        self.n_actions = n_actions
        self.memory = []  # 经验回放
        self.memory_size = 10000
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_target_freq = 1000
        
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
    
    def act(self, state):
        """选择动作"""
        if np.random.random() <= self.epsilon:
            return np.random.randint(self.n_actions)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(self.device)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(self.device)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(self.device)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(self.device)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

### 3. 策略梯度方法

**思想：** 直接优化策略参数

```python
class PolicyGradientAgent:
    def __init__(self, input_dim, n_actions):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 策略网络：输入状态，输出动作概率
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=0.001)
        
    def act(self, state):
        """根据策略采样动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state_tensor).squeeze(0)
        action = torch.multinomial(action_probs, 1).item()
        return action, action_probs[action].item()
    
    def update(self, states, actions, rewards):
        """策略梯度更新"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # 计算动作概率
        action_probs = self.policy_network(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算损失 (REINFORCE 算法)
        log_probs = torch.log(selected_action_probs + 1e-8)
        loss = -(log_probs * rewards).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

## 🎮 决策智能的应用

### 1. 游戏 AI

**Atari 游戏示例：**
```python
"""
DeepMind 的突破：
- 使用原始像素输入
- CNN 提取视觉特征
- DQN 学习游戏策略
- 在多个 Atari 游戏上超越人类
"""
```

### 2. 机器人控制

**连续控制示例：**
```python
"""
连续动作空间：
- 传统的 Q-Learning 不适用
- 需要 Actor-Critic 方法
- 策略梯度算法
"""
```

### 3. 自动驾驶

**决策流程：**
```
传感器数据 → 环境感知 → 路径规划 → 行为决策 → 轨迹跟踪
```

## 🔬 智能决策的关键技术

### 1. 探索策略

#### Upper Confidence Bound (UCB)
```python
def ucb_action_selection(action_values, action_counts, total_plays, c=2):
    """UCB 算法选择动作"""
    ucb_values = action_values + c * np.sqrt(np.log(total_plays) / (action_counts + 1e-8))
    return np.argmax(ucb_values)
```

#### Thompson Sampling
```python
def thompson_sampling(successes, failures):
    """汤普森采样"""
    samples = np.random.beta(successes + 1, failures + 1)
    return np.argmax(samples)
```

### 2. 经验回放

**目的：** 打破数据相关性，提高样本效率

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
```

### 3. 目标网络

**目的：** 稳定训练过程

```python
def update_target_network(main_network, target_network):
    """更新目标网络"""
    target_network.load_state_dict(main_network.state_dict())

# 在训练中定期更新
if step % target_update_freq == 0:
    update_target_network(q_network, target_network)
```

## 💡 智能决策的挑战与解决方案

### 1. 信用分配问题

**问题：** 如何将长期回报分配给各个动作

**解决方案：**
- **资格迹 (Eligibility Traces)**: TD(λ) 算法
- **优势函数**: A3C、A2C
- **Actor-Critic**: 结合值函数和策略梯度

### 2. 探索效率问题

**问题：** 如何在未知环境中有效探索

**解决方案：**
- **好奇心驱动**: Intrinsic Motivation
- **奖励塑形**: Reward Shaping
- **层次化学习**: Hierarchical RL

### 3. 迁移学习

**问题：** 如何将在一个任务上学到的知识应用到新任务

**解决方案：**
- **预训练**: Pre-trained representations
- **元学习**: Learning to Learn
- **多任务学习**: Multi-task learning

## 🎓 学习要点总结

### 从试错到智能的演进

1. **朴素试错**
   - 随机行为
   - 无记忆
   - 效率低下

2. **带记忆试错**
   - 记录经验
   - 平均回报
   - 简单优化

3. **智能决策**
   - 状态感知
   - 策略优化
   - 长期规划

### 关键技术

1. **值函数方法**
   - Q-Learning
   - SARSA
   - Deep Q-Network

2. **策略梯度方法**
   - REINFORCE
   - Actor-Critic
   - A3C/A2C

3. **探索策略**
   - ε-贪婪
   - UCB
   - Thompson Sampling

### 实践要点

1. **算法选择**
   - 离散动作: Q-Learning, DQN
   - 连续动作: Policy Gradient, Actor-Critic
   - 大状态空间: 函数逼近

2. **超参数调优**
   - 学习率
   - 折扣因子
   - 探索率

3. **稳定性技巧**
   - 经验回放
   - 目标网络
   - 梯度裁剪

## 🚀 下一步

现在我们理解了从试错到智能决策的演进过程，接下来让我们深入了解 Q-Learning 算法。

---

**下一步：** [Day26-Q3 - Q-Learning 基础](./Day26-Q3%20-%20Q-Learning%20基础.md)
---

## 📱 关于作者 & 获取更多资源

本教程由 **Lee（职场宝爸）** 创建，记录从零基础到独立完成 AI 项目的真实历程。

### 关注公众号，获取独家内容

**公众号名称：Lee 的成长日记**

微信搜索关注，获取：
- ✅ **AI 学习路线规划**：零基础如何系统学习 AI
- ✅ **项目实战源码**：完整可运行的项目代码
- ✅ **深度技术解析**：前沿技术原理 + 手写代码实现
- ✅ **职场成长心得**：一个宝爸的 AI 逆袭之路

**关注福利**：
- 回复「**路线**」→ 获取 30 天 AI 学习计划表
- 回复「**项目**」→ 获取 GitHub 项目源码合集
- 回复「**资料**」→ 获取零基础学习资源推荐

**扫码关注公众号**：

![公众号二维码](../../images/logos/ewm.jpg)

### 其他平台

- 📂 **GitHub**：https://github.com/Lee985-cmd/AI-30Days-Challenge
- 📝 **CSDN 博客**：https://blog.csdn.net/m0_67081842
- 💬 **公众号**：微信搜索「Lee 的成长日记」

---

> 💡 **学习建议**
> 
> 如果本篇教程对你有帮助，欢迎：
> 1. **Star GitHub 项目**：https://github.com/Lee985-cmd/AI-30Days-Challenge
> 2. **关注公众号**获取更多独家内容
> 3. **留言交流**你的学习困惑
> 
> **一起学习，一起进步！** 🤝
