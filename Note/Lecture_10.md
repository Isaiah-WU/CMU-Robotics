# 机器人学习（十）：策略梯度（下）与 Actor-Critic

> 对应 CMU 16-831 Lecture 10：Policy Gradient Methods (Part II) and Actor-Critic Methods。
> 读法提示：每个重要公式下面都配了一段"**读式子**"，把数学翻译成人话；符号不认识就翻回 2.1 的速查表。

## 1. 本讲主线

一句话概括：**策略梯度 (policy gradient) 想法漂亮，但方差太大、样本太费。本讲沿着"降方差"和"省样本"两条线，从 REINFORCE 一路推进到 actor-critic，再到 off-policy actor-critic，最后落到 SAC、DDPG 两个实用算法。**

```mermaid
flowchart LR
    A["REINFORCE<br>蒙特卡洛策略梯度<br>问题：方差大"] --> B["降方差<br>因果性 causality<br>+ 基线 baseline"]
    B --> C["Actor-Critic<br>critic 网络估计优势<br>问题：仍是 on-policy"]
    C --> D["Off-policy AC<br>replay buffer<br>+ 动作重采样"]
    D --> E["实用算法<br>SAC · DDPG"]

    classDef step fill:#EDF3FB,stroke:#7FA7D1,color:#20416B
    classDef final fill:#EAF6EE,stroke:#7DBE8E,color:#1F4D2E
    class A,B,C,D step
    class E final
```

两条暗线值得先记住：

- **降方差 (variance reduction)**：$R(\tau)$ → reward-to-go → 减基线 (baseline) → 优势函数 (advantage function) → 自举 (bootstrap)。每一步都在让梯度里的权重"更瘦、更稳"。
- **省样本 (sample efficiency)**：on-policy → 重要性采样 (importance sampling) → 并行采样 (parallel sampling) → 经验回放 (replay buffer)。

## 2. 复习：策略梯度 (Policy Gradient, PG)

### 2.1 符号速查表

先认符号，再读句子。全篇的记号都在这张表里：

| 符号 | 读法 | 含义 |
|---|---|---|
| $s,\ a,\ r$ | | 状态 (state)、动作 (action)、奖励 (reward) |
| $\tau$ | tau | 一条轨迹 (trajectory)：$(s_1,a_1,s_2,a_2,\dots)$。第 10 节的软更新系数也用这个字母，纯属撞车 |
| $\pi_\theta(a\mid s)$ | pi | 策略：在状态 $s$ 下选动作 $a$ 的概率；$\theta$ 是策略网络（actor）的参数 |
| $p(s'\mid s,a)$ | | 环境动力学 (dynamics)：在 $s$ 做 $a$ 后转到 $s'$ 的概率，机器人并不知道它 |
| $E_{x\sim p}[\cdot]$ | 期望 | 按分布 $p$ 抽 $x$，对方括号里的量取平均 |
| $\nabla_\theta$ | 梯度 | 对参数 $\theta$ 求导：指向"让目标涨得最快"的参数调整方向 |
| $J(\theta)$ | | 目标函数：用参数 $\theta$ 的策略平均能拿到的总回报 |
| $Q^\pi(s,a)$ | Q 值 | 动作价值：在 $s$ 先做 $a$、之后一直按 $\pi$ 行动的期望总回报 |
| $V^\pi(s)$ | V 值 | 状态价值：在 $s$ 一直按 $\pi$ 行动的期望总回报（第一步动作也被平均掉） |
| $A^\pi(s,a)$ | 优势 | $Q^\pi(s,a)-V^\pi(s)$：这个动作比该状态的平均水平好多少 |
| $\gamma$ | gamma | 折扣因子 (discount factor)，$0<\gamma<1$，越远的奖励越打折 |
| $\alpha$ | alpha | 学习率 (learning rate)；在 SAC 一节另指温度系数 |
| $\theta,\ \phi$ | theta, phi | 分别是 actor（策略）与 critic（价值网络）的参数 |
| $\hat Q,\ \hat V,\ \hat A$ | hat（帽子） | 戴帽子的都是**估计值**，不是真值 |
| $x\sim p$ | 采样自 | $x$ 按分布 $p$ 抽出，如 $a\sim\pi_\theta$ 表示动作由策略采样 |
| $\propto$ | 正比于 | 两边差一个常数倍，不影响梯度的方向 |

### 2.2 目标函数

策略 (policy) $\pi_\theta$ 与环境动力学 (dynamics) $p(s'\mid s,a)$ 共同决定一条轨迹 (trajectory) $\tau=(s_1,a_1,\dots,s_T,a_T)$ 的分布，即轨迹似然 (trajectory likelihood)：

$$p_\theta(\tau)=p(s_1)\prod_{t=1}^{T}\pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)$$

> **读式子**：右边把"跑出这条轨迹"的概率按时间拆开——先按 $p(s_1)$ 抽初始状态；之后每一步发生两件事：策略掷骰子选动作（$\pi_\theta(a_t\mid s_t)$），环境掷骰子给下一个状态（$p(s_{t+1}\mid s_t,a_t)$）；把每步的概率连乘（$\prod$），就是整条轨迹的概率。下标 $\theta$ 提醒：策略一变，轨迹的分布跟着变。

目标是最大化期望回报 (expected return)：

$$\theta^\star=\arg\max_\theta J(\theta),\qquad J(\theta)=E_{\tau\sim p_\theta(\tau)}\big[R(\tau)\big],\qquad R(\tau)=\sum_t r(s_t,a_t)$$

> **读式子**：$R(\tau)$ 是这条轨迹攒下的总奖励，叫回报 (return)。$E_{\tau\sim p_\theta(\tau)}[\cdot]$ 读作"让策略反复去跑、对所有可能跑出的轨迹取平均"。所以 $J(\theta)$ 就是"参数为 $\theta$ 的策略平均拿多少分"；$\arg\max_\theta$ 是找让平均分最高的那组参数。

策略梯度的思路最直接：不显式学值函数、不建环境模型，直接对 $J(\theta)$ 做梯度上升 (gradient ascent)，也就是要算 $\nabla_\theta J(\theta)$。

### 2.3 策略梯度定理 (policy gradient theorem)

$\nabla_\theta J(\theta)$ 难算的原因：$\theta$ 同时影响两件事——每一步选什么动作，以及因此会到达哪些状态（状态分布 (state distribution) 也依赖 $\theta$）。直接求导会撞上未知的动力学。策略梯度定理把它绕开了：

$$\nabla_\theta J(\theta)=E_{\tau\sim p_\theta(\tau)}\Big[R(\tau)\sum_t\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big]$$

> **读式子**：$\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 是"往哪个方向调 $\theta$，能让『在 $s_t$ 选 $a_t$』的概率变大"——取对数是为了连乘变连加、求导好算，方向不变。$\sum_t$ 把整条轨迹每一步的这种方向加起来，再乘整条轨迹的得分 $R(\tau)$：得分高就顺着走（这些动作的概率统统调大），得分为负就反着走。最外层期望表示对多条轨迹取平均。妙处：式子里只剩策略 $\pi_\theta$，环境动力学 $p(s'\mid s,a)$ 消失了——不需要环境模型也能估梯度，这是 model-free 的根源。

直觉：形式上像"按回报加权的最大似然