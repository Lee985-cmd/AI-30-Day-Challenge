# Day26-Q3 - Q-Learning 基础

## 🎯 Q-Learning 算法详解

### 什么是 Q-Learning？

**Q-Learning** 是一种无模型的强化学习算法，用于学习一个最优策略。它通过学习一个 Q 函数（动作-价值函数）来实现，该函数预测在给定状态下采取某个动作的长期回报。

**Q 函数的含义：**
```
Q(s, a) = 在状态 s 下采取动作 a 的预期累积回报
```

### Q-Learning 的核心思想

```python
"""
Q-Learning 的哲学:

1. 试错学习
   - 尝试不同的动作
   - 观察结果
   - 更新认识

2. 贪婪优化
   - 总是寻找最优动作
   - 但也需要探索

3. Bellman 方程
   - 今天的决策 = 明天的预期 + 今天的奖励
   - Q(s,a) = r + γ * max Q(s',a')
"""
```

### Q-Learning 算法详解

#### 1. Q 函数更新规则

**Bellman 方程：**
```
Q(s,a) ← Q(s,a) + α[r + γ * max Q(s',a') - Q(s,a)]
                              a'

其中:
- s: 当前状态
- a: 当前动作
- r: 获得的奖励
- s': 下一状态
- α: 学习率
- γ: 折扣因子
- max Q(s',a'): 下一状态的最大 Q 值
```

**直观理解：**
```
新 Q 值 = 旧 Q 值 + 学习率 × (目标值 - 旧 Q 值)
                    ↑
                 时序差分误差 (TD Error)

目标值 = 即时奖励 + 未来价值
```

#### 2. 完整算法流程

```python
def q_learning_algorithm(env, episodes=1000):
    """
    Q-Learning 算法实现
    
    参数:
    env: 环境对象
    episodes: 训练回合数
    """
    
    # 初始化 Q 表 (状态数 × 动作数)
    q_table = np.zeros((env.n_states, env.n_actions))
    
    # 超参数
    learning_rate = 0.1      # α
    discount_factor = 0.95   # γ
    epsilon = 1.0           # 探索率
    epsilon_decay = 0.995   # 探索衰减
    epsilon_min = 0.01      # 最小探索率
    
    for episode in range(episodes):
        # 重置环境
        state = env.reset()
        total_reward = 0
        
        done = False
        while not done:
            # ε-贪婪策略选择动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()  # 随机探索
            else:
                action = np.argmax(q_table[state])  # 贪婪利用
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # Q-Learning 更新
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + discount_factor * best_next_action
            td_error = td_target - q_table[state][action]
            q_table[state][action] += learning_rate * td_error
            
            # 更新状态
            state = next_state
            total_reward += reward
        
        # 探索率衰减
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # 打印进度
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")
    
    return q_table
```

### Q-Learning 的关键特性

#### 1. 离策略学习 (Off-policy)

**含义：** 学习的策略与行为的策略不同

```python
"""
离策略 vs 在策略:

在策略 (On-policy):
- 学习 π 时也使用 π 与环境交互
- 例子: SARSA

离策略 (Off-policy):
- 学习 π 时使用不同的策略 μ 与环境交互
- 例子: Q-Learning

Q-Learning 的离策略特性:
- 行为策略: ε-贪婪 (探索)
- 目标策略: 贪婪 (利用)
- 优势: 探索和利用分离
"""
```

#### 2. 收敛性保证

**理论保证：**
- 在满足一定条件下，Q-Learning 收敛到最优 Q 函数
- 条件：所有状态-动作对被无限访问

```python
"""
收敛条件:
1. 所有状态-动作对被访问无限次
2. 学习率满足 Robbins-Monro 条件:
   - Σα = ∞ (学习率总和无穷)
   - Σα² < ∞ (学习率平方和有限)
   
实践中:
- 逐渐减小探索率
- 足够的训练步数
- 合适的学习率
"""
```

### Q-Learning 实例：Grid World

让我们通过一个具体的例子来理解 Q-Learning：

```python
import numpy as np
import random

class GridWorld:
    """简单的网格世界环境"""
    
    def __init__(self):
        self.width = 4
        self.height = 4
        self.n_states = self.width * self.height
        self.n_actions = 4  # 上、下、左、右
        
        # 目标位置 (奖励 +10)
        self.goal = (3, 3)
        
        # 障碍位置
        self.obstacles = [(1, 1), (2, 2)]
        
        # 当前位置
        self.position = (0, 0)
        
    def reset(self):
        """重置环境"""
        self.position = (0, 0)
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态 (转换为一维)"""
        row, col = self.position
        return row * self.width + col
    
    def _get_position_from_state(self, state):
        """从状态恢复位置"""
        row = state // self.width
        col = state % self.width
        return (row, col)
    
    def step(self, action):
        """执行动作"""
        row, col = self.position
        
        # 动作: 0=上, 1=下, 2=左, 3=右
        if action == 0:  # 上
            row = max(0, row - 1)
        elif action == 1:  # 下
            row = min(self.height - 1, row + 1)
        elif action == 2:  # 左
            col = max(0, col - 1)
        elif action == 3:  # 右
            col = min(self.width - 1, col + 1)
        
        # 检查障碍
        if (row, col) in self.obstacles:
            # 如果撞到障碍，保持原位
            pass
        else:
            self.position = (row, col)
        
        # 计算奖励
        if self.position == self.goal:
            reward = 10  # 到达目标
            done = True
        elif (row, col) in self.obstacles:
            reward = -1  # 撞到障碍
            done = False
        else:
            reward = -0.1  # 每步小惩罚 (鼓励快速到达)
            done = False
        
        return self._get_state(), reward, done, {}

# Q-Learning 训练
def train_grid_world():
    env = GridWorld()
    
    # 初始化 Q 表
    q_table = np.zeros((env.n_states, env.n_actions))
    
    # 超参数
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    
    episodes = 1000
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        done = False
        while not done and steps < 100:  # 限制步数
            # ε-贪婪选择动作
            if random.random() < epsilon:
                action = random.randint(0, env.n_actions - 1)
            else:
                action = np.argmax(q_table[state])
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # Q-Learning 更新
            best_next_q = np.max(q_table[next_state])
            current_q = q_table[state, action]
            td_error = reward + discount_factor * best_next_q - current_q
            q_table[state, action] += learning_rate * td_error
            
            state = next_state
            total_reward += reward
            steps += 1
        
        # ε 衰减
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # 打印进度
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {epsilon:.3f}")
    
    return q_table

# 训练
q_table = train_grid_world()
print("训练完成!")
```

### Q-Learning 的变种

#### 1. Double Q-Learning

**问题：** Q-Learning 有正向偏差（overestimation）

```python
"""
正向偏差问题:

Q-Learning:
Q(s,a) ← r + γ * max Q(s',a')

max 操作总是选择最大值
即使 Q 值估计不准，也会选择最大值
导致 Q 值被高估

Double Q-Learning 解决方案:
维护两个 Q 表: Q1, Q2
更新时使用其中一个选择动作，另一个评估价值

Q1(s,a) ← r + γ * Q2(s', argmax Q1(s',a'))
Q2(s,a) ← r + γ * Q1(s', argmax Q2(s',a'))
"""
```

**实现：**
```python
class DoubleQLearning:
    def __init__(self, n_states, n_actions):
        self.q1 = np.zeros((n_states, n_actions))
        self.q2 = np.zeros((n_states, n_actions))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
    
    def get_action(self, state, epsilon=0.1):
        """ε-贪婪选择动作"""
        if np.random.random() < epsilon:
            return np.random.randint(len(self.q1[state]))
        
        # 使用平均 Q 值选择动作
        avg_q = (self.q1[state] + self.q2[state]) / 2
        return np.argmax(avg_q)
    
    def update(self, state, action, reward, next_state):
        """更新 Q 表"""
        if np.random.random() < 0.5:
            # 更新 Q1
            best_next_action = np.argmax(self.q1[next_state])
            td_target = reward + self.discount_factor * self.q2[next_state][best_next_action]
            self.q1[state][action] += self.learning_rate * (td_target - self.q1[state][action])
        else:
            # 更新 Q2
            best_next_action = np.argmax(self.q2[next_state])
            td_target = reward + self.discount_factor * self.q1[next_state][best_next_action]
            self.q2[state][action] += self.learning_rate * (td_target - self.q2[state][action])
```

#### 2. Dueling DQN

**思想：** 将 Q 值分解为状态价值和优势函数

```python
"""
Dueling DQN:

Q(s,a) = V(s) + A(s,a)

其中:
- V(s): 状态价值 (state value)
- A(s,a): 动作优势 (advantage)

优势函数:
A(s,a) = Q(s,a) - V(s)

网络结构:
输入 → 共享层 → 
        ├─ 状态价值分支 (V)
        └─ 优势函数分支 (A)
        
最终输出: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
```

**优势：**
- 更好的价值估计
- 加速收敛
- 更稳定的训练

### Q-Learning 的局限性

#### 1. 状态空间限制

```python
"""
问题:
- Q-Learning 需要为每个状态-动作对维护 Q 值
- 对于连续状态空间，Q 表无法存储
- 组合爆炸问题

解决方案:
- 函数逼近 (Function Approximation)
- 深度学习 (Deep Q-Network)
- 特征工程
"""
```

#### 2. 连续动作空间

```python
"""
问题:
- Q-Learning 只适用于离散动作空间
- 无法直接处理连续动作

解决方案:
- 离散化动作空间
- 策略梯度方法 (Policy Gradients)
- Actor-Critic 方法
"""
```

#### 3. 探索效率

```python
"""
问题:
- ε-贪婪探索效率不高
- 在大型状态空间中难以充分探索

解决方案:
- 好奇心驱动 (Intrinsic Motivation)
- 奖励塑形 (Reward Shaping)
- 分层强化学习 (Hierarchical RL)
"""
```

### Q-Learning 的实际应用

#### 1. 游戏 AI

```python
"""
Atari 游戏示例:

状态: 游戏画面 (像素)
动作: 方向键 + 按钮
奖励: 游戏得分
目标: 最大化累计得分

挑战:
- 状态空间巨大 (图像)
- 稀疏奖励
- 部分可观测
"""
```

#### 2. 机器人控制

```python
"""
机械臂控制:

状态: 关节角度、速度
动作: 力矩、速度
奖励: 到达目标 +1，碰撞 -1
目标: 高效精准控制

挑战:
- 连续动作空间
- 安全约束
- 实时性要求
"""
```

#### 3. 推荐系统

```python
"""
内容推荐:

状态: 用户特征、上下文
动作: 推荐内容
奖励: 点击、观看时长
目标: 最大化用户参与度

挑战:
- 冷启动问题
- 长期价值
- 多目标优化
"""
```

### Q-Learning 与其他算法对比

| 算法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| Q-Learning | 离散状态/动作 | 简单、收敛性好 | 维度限制 |
| SARSA | 在策略学习 | 稳定、安全 | 可能陷入局部最优 |
| DQN | 大状态空间 | 深度学习能力 | 训练不稳定 |
| Policy Gradient | 连续动作 | 直接优化策略 | 方差大 |

## 🎓 学习要点总结

### Q-Learning 核心

1. **算法原理**
   - Bellman 方程
   - 时序差分学习
   - 离策略学习

2. **更新规则**
   - Q(s,a) ← Q(s,a) + α[r + γmaxQ(s',a') - Q(s,a)]
   - 目标值 vs 当前值
   - TD 误差

3. **探索策略**
   - ε-贪婪
   - 收敛保证
   - 探索 vs 利用

### 关键技术

1. **收敛性**
   - 理论保证
   - 实践条件
   - 超参数设置

2. **变种算法**
   - Double Q-Learning (消除偏差)
   - Dueling DQN (价值分解)
   - Prioritized Replay (重要经验)

3. **应用领域**
   - 游戏 AI
   - 机器人控制
   - 推荐系统

### 局限与改进

1. **状态空间**
   - 传统: 离散有限
   - 改进: 函数逼近
   - 现代: 深度学习

2. **动作空间**
   - 传统: 离散
   - 改进: 策略梯度
   - 现代: Actor-Critic

3. **探索效率**
   - 传统: ε-贪婪
   - 改进: 好奇心驱动
   - 现代: 奖励塑形

## 🚀 下一步

现在我们深入理解了 Q-Learning 算法，接下来让我们探索 Deep Q-Network (DQN)，这是将深度学习与 Q-Learning 结合的重要突破。

---

**下一步：** [Day26-Q4 - Deep Q-Network(DQN)](./Day26-Q4%20-%20Deep%20Q-Network%28DQN%29.md)