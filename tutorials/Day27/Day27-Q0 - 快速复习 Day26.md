# Day27-Q0 - 快速复习 Day26

## 🔄 强化学习要点回顾

### 核心概念速记

**强化学习三要素:**
```
Agent (智能体) ←→ Environment (环境)
     ↓                    ↓
  Action (动作)      State + Reward
```

**关键公式:**
- **Bellman 方程**: V(s) = E[r + γV(s')]
- **Q-Learning 更新**: Q(s,a) ← Q(s,a) + α[r + γmax Q(s',a') - Q(s,a)]
- **折扣回报**: G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...

---

## 📝 Day26 知识点检查

### Q1: 强化学习基础
- [ ] 能解释 Agent、Environment、State、Action、Reward 的关系
- [ ] 理解 MDP (马尔可夫决策过程) 的五个要素
- [ ] 知道强化学习与监督学习的区别

### Q2: 探索与利用
- [ ] 掌握 ε-greedy 策略的工作原理
- [ ] 理解 UCB (Upper Confidence Bound) 的思想
- [ ] 知道为什么需要平衡探索和利用

### Q3: Q-Learning
- [ ] 能写出 Bellman 方程
- [ ] 理解 Q 表的更新规则
- [ ] 知道 Q-Learning 是 off-policy 算法

### Q4: Deep Q-Network
- [ ] 理解为什么需要经验回放 (Experience Replay)
- [ ] 知道目标网络 (Target Network) 的作用
- [ ] 能用 PyTorch 实现简单的 DQN

### Q5: Flappy Bird AI
- [ ] 会用 Gym 框架创建自定义环境
- [ ] 理解奖励函数设计的重要性
- [ ] 知道如何调试和优化 RL 训练

### Q6: 进阶算法
- [ ] 了解 Policy Gradient 的基本思想
- [ ] 知道 Actor-Critic 架构的优势
- [ ] 听说过 PPO、SAC 等现代算法

---

## 💻 代码回顾

### DQN 核心代码

```python
# 1. 选择动作 (ε-greedy)
def select_action(self, state):
    if random.random() < self.epsilon:
        return random.randint(0, self.action_dim - 1)
    else:
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state))
            return q_values.argmax().item()

# 2. 存储经验
def store_transition(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

# 3. 训练网络
def train(self):
    # 采样批次
    batch = random.sample(self.memory, self.batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)
    
    # 计算当前 Q 值
    current_q = self.policy_net(states).gather(1, actions)
    
    # 计算目标 Q 值 (使用目标网络)
    with torch.no_grad():
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)
    
    # 反向传播
    loss = nn.MSELoss()(current_q, target_q)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

---

## 🎯 从 Day26 到 Day27 的过渡

**Day26 我们学会了:**
- ✅ 训练 AI 模型 (Flappy Bird AI)
- ✅ 在本地运行和测试
- ✅ 调整超参数优化性能

**Day27 我们要学习:**
- 🚀 如何把训练好的模型部署到服务器
- 🚀 如何让其他人通过 API 调用你的模型
- 🚀 如何监控和维护生产环境的模型
- 🚀 如何优化模型性能和降低成本

**类比:**
```
Day26: 在实验室里研发出新药
   ↓
Day27: 把药厂建起来,批量生产,卖给医院
```

---

## 🔗 相关链接

- [← Day26-Q6 - 强化学习进阶](./Day26-Q6%20-%20强化学习进阶与未来展望.md)
- [→ Day27-Q1 - 为什么要部署模型?](./Day27-Q1%20-%20为什么要部署模型？.md)

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

![公众号二维码](../../../images/logos/ewm.jpg)

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
