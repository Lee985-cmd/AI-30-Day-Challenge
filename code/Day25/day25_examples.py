"""
Day25 - 代码示例

本文件从教程文档中自动提取，包含可运行的代码示例。

运行方法:
    python day25_examples.py

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
print("Day25 - 代码示例")
print("=" * 60)
print()


# ===== 代码块 1 =====

"""
Q-learning 算法步骤:

初始化 Q-table（全 0 或随机）

重复以下步骤直到学会:

1. 观察当前状态 s

2. 选择动作 a
   - 探索 (Exploration): 随机选一个动作
   - 利用 (Exploitation): 选 Q 值最大的动作
   
   为什么要探索？
   → 可能更好的动作还没试过
   → 不能只盯着已知的

3. 执行动作 a，得到奖励 r 和新状态 s'

4. 更新 Q-table:
   Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s')) - Q(s,a)]
   
   解释每一项:
   - α (alpha): 学习率 (0-1)
     → 学多快？1 = 完全相信新经验，0 = 不学习
   
   - γ (gamma): 折扣因子 (0-1)
     → 多看重未来？1 = 只看长远，0 = 只顾眼前
   
   - r: 即时奖励
     → 这次做得怎么样？
   
   - max(Q(s')): 未来最大期望
     → 下一步最多能得多少？

5. 进入新状态 s'，继续循环

最后:
Q-table 收敛 → 学会最优策略
"""

# ===== 代码块 2 =====

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("🎮 Q-learning 从零实现 - 股票交易示例")
print("=" * 60)

class StockTradingEnv:
    """简化的股票交易环境"""
    
    def __init__(self, initial_price=100):
        self.initial_price = initial_price
        self.price = initial_price
        self.day = 0
        self.prices = []
        
        # 生成一些模拟股价数据
        np.random.seed(42)
        days = 100
        changes = np.random.randn(days).cumsum() * 2  # 随机游走
        self.all_prices = self.initial_price + changes
        
    def reset(self):
        """重置环境"""
        self.price = self.initial_price
        self.day = 0
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        # 简化：状态就是当前股价的区间
        if self.price < 90:
            return 0  # 低价区
        elif self.price < 110:
            return 1  # 中价区
        else:
            return 2  # 高价区
    
    def step(self, action, shares=100):
        """
        执行动作
        
        参数:
        action: 0=买入，1=卖出，2=持有
        shares: 交易股数
        
        返回:
        new_state: 新状态
        reward: 奖励
        done: 是否结束
        """
        old_price = self.price
        
        # 进入下一天
        self.day += 1
        if self.day >= len(self.all_prices):
            self.day = len(self.all_prices) - 1
        
        self.price = self.all_prices[self.day]
        price_change = self.price - old_price
        
        # 计算奖励
        if action == 0:  # 买入
            reward = price_change * shares  # 股价上涨赚钱
        elif action == 1:  # 卖出
            reward = -price_change * shares  # 股价下跌赚钱（做空）
        else:  # 持有
            reward = 0
        
        new_state = self._get_state()
        done = self.day >= len(self.all_prices) - 1
        
        return new_state, reward, done

class QLearningAgent:
    """Q-learning 智能体"""
    
    def __init__(self, n_states=3, n_actions=3, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        参数:
        n_states: 状态数
        n_actions: 动作数
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 探索率（ε-greedy 策略）
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        
        # 初始化 Q-table (状态数 × 动作数)
        self.q_table = np.zeros((n_states, n_actions))
        
        print(f"\n✓ Q-learning Agent 初始化完成")
        print(f"  状态数：{n_states}")
        print(f"  动作数：{n_actions}")
        print(f"  学习率 (α): {alpha}")
        print(f"  折扣因子 (γ): {gamma}")
        print(f"  探索率 (ε): {epsilon}")
    
    def choose_action(self, state):
        """选择动作（ε-greedy 策略）"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择 Q 值最大的动作
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """更新 Q-table"""
        if not done:
            # Q-learning 更新公式
            best_next_action = np.argmax(self.q_table[next_state])
            td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
            td_error = td_target - self.q_table[state][action]
            self.q_table[state][action] += self.alpha * td_error
        else:
            # 终止状态，没有未来
            self.q_table[state][action] += self.alpha * (reward - self.q_table[state][action])
    
    def get_policy(self):
        """获取策略（从 Q-table 提取）"""
        policy = np.argmax(self.q_table, axis=1)
        return policy

# ============================================================================
# 训练过程
# ============================================================================

print("\n" + "=" * 60)
print("【开始训练】Q-learning 股票交易策略")
print("=" * 60)

# 创建环境和智能体
env = StockTradingEnv(initial_price=100)
agent = QLearningAgent(n_states=3, n_actions=3, alpha=0.1, gamma=0.9, epsilon=0.1)

# 训练参数
n_episodes = 1000
max_steps = 50

# 记录奖励历史
rewards_history = []
avg_rewards = []

print(f"\n训练设置:")
print(f"  - 训练轮数：{n_episodes}")
print(f"  - 每轮最大步数：{max_steps}")
print(f"  - 初始探索率：{agent.epsilon}")

print("\n开始训练...\n")

for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # 选择动作
        action = agent.choose_action(state)
        
        # 执行动作
        next_state, reward, done = env.step(action)
        
        # 更新 Q-table
        agent.update(state, action, reward, next_state, done)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    rewards_history.append(total_reward)
    
    # 计算移动平均
    if len(rewards_history) >= 10:
        avg_reward = np.mean(rewards_history[-10:])
        avg_rewards.append(avg_reward)
    
    # 显示进度
    if (episode + 1) % 100 == 0:
        avg = np.mean(rewards_history[-100:])
        print(f"Episode {episode+1}/{n_episodes} - 平均奖励：{avg:.2f}")

print("\n✓ 训练完成!")

# ============================================================================
# 可视化结果
# ============================================================================

print("\n" + "=" * 60)
print("【结果可视化】")
print("=" * 60)

# 绘制奖励变化
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(rewards_history, alpha=0.6, label='单轮奖励')
if len(avg_rewards) > 0:
    plt.plot(range(10, len(rewards_history)+1), avg_rewards, 'r-', linewidth=2, label='10 轮平均')
plt.xlabel('训练轮数')
plt.ylabel('累积奖励')
plt.title('训练过程中的奖励变化')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.imshow(agent.q_table, cmap='Blues', aspect='auto')
plt.colorbar(label='Q 值')
plt.xticks(range(3), ['买入', '卖出', '持有'])
plt.yticks(range(3), ['低价区', '中价区', '高价区'])
plt.title('训练后的 Q-table')

# 在格子中显示数值
for i in range(3):
    for j in range(3):
        plt.text(j, i, f'{agent.q_table[i, j]:.2f}',
                ha='center', va='center', color='black' if agent.q_table[i, j] < 0.5 else 'white')

plt.tight_layout()
plt.show()

# ============================================================================
# 展示学到的策略
# ============================================================================

print("\n" + "=" * 60)
print("【学到的交易策略】")
print("=" * 60)

policy = agent.get_policy()
action_names = ['买入', '卖出', '持有']
state_names = ['低价区 (<90)', '中价区 (90-110)', '高价区 (>110)']

print("\n在不同状态下的最优动作:")
for state in range(3):
    action = policy[state]
    q_values = agent.q_table[state]
    best_q = q_values[action]
    
    emoji = {'买入': '🟢', '卖出': '🔴', '持有': '⚪'}
    
    print(f"\n{state_names[state]}:")
    print(f"  最优动作：{emoji[action_names[action]]} {action_names[action]}")
    print(f"  Q 值：{best_q:.3f}")
    print(f"  所有动作 Q 值：买入={q_values[0]:.3f}, 卖出={q_values[1]:.3f}, 持有={q_values[2]:.3f}")

print("\n💡 策略解读:")
if policy[0] == 0:  # 低价区买入
    print("  ✓ 低价区买入 - 价值投资理念")
if policy[2] == 1:  # 高价区卖出
    print("  ✓ 高价区卖出 - 止盈策略")
if policy[1] == 2:  # 中价区持有
    print("  ✓ 中价区持有 - 观望等待")

print("\n🎊 Q-learning 训练完成!")
print("=" * 60)

# ============================================================================
# 实际应用建议
# ============================================================================

print("\n" + "=" * 60)
print("【6. 实际应用建议】")
print("=" * 60)

print("""
使用场景推荐:

1. 自动化交易:
   ✓ 学习历史交易数据
   ✓ 发现盈利模式
   ✓ 自动执行买卖
   ✓ 24 小时监控市场

2. 游戏 AI:
   ✓ AlphaGo 下围棋
   ✓ Dota2 游戏 AI
   ✓ 超级玛丽通关
   ✓  Atari 游戏大师

3. 机器人控制:
   ✓ 机械臂抓取
   ✓ 自动驾驶
   ✓ 无人机飞行
   ✓ 人形机器人走路

4. 资源调度:
   ✓ 数据中心冷却
   ✓ 交通信号控制
   ✓ 电力网络优化
   ✓ 物流配送路径

技术要点:

✓ 状态空间设计
  - 不能太大（计算不了）
  - 不能太小（信息不足）
  - 可以用函数近似（神经网络）

✓ 奖励函数设计
  - 要及时（延迟奖励难学）
  - 要准确（反映真实目标）
  - 要平衡短期和长期

✓ 探索 vs 利用
  - ε-greedy: 简单有效
  - UCB: 更智能的探索
  - Thompson Sampling: 概率方法

常见问题:

✗ 维度灾难
  → 状态太多，Q-table 存不下
  → 解决：用 DQN（深度 Q 网络）

✗ 奖励稀疏
  → 很久才得到一次奖励
  → 解决：设计中间奖励

✗ 不收敛
  → 学习率太高或太低
  → 调整α、γ、ε参数
""")

print("\n🎉 强化学习入门完成!")
print("=" * 60)

# ===== 代码块 3 =====

"""
DQN = Deep Q-Network（深度 Q 网络）

解决的问题:
Q-learning 只能处理小状态空间
→ 状态多了，Q-table 太大存不下
→ 比如围棋：10^170 种状态
→ 比如图像：每个像素都是状态

解决方案:
用神经网络代替 Q-table!

输入：状态（比如游戏画面）
      ↓
   神经网络
      ↓
输出：每个动作的 Q 值

好处:
✓ 可以处理高维状态
✓ 可以泛化到未见过的状态
✓ AlphaGo、Dota2 AI 都用这个

但 DQN 很复杂，需要:
- 经验回放（Experience Replay）
- 目标网络（Target Network）
- 梯度裁剪（Gradient Clipping）

这些留给进阶课程！
"""