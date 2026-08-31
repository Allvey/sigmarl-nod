# BF-MARL：分岔形式多智能体强化学习可移植技术方案

> 文档目标：给出一套不依赖特定仓库、仿真器或强化学习框架的实现规范。  
> 适用问题：多机器人会车、无信号交叉口、多车汇入、窄通道通行、任务分配等包含“未决—模式选择—承诺保持”的连续控制任务。  
> 推荐名称：**Bifurcation-Form Multi-Agent Reinforcement Learning（BF-MARL）**。

---

## 1. 方法定位

BF-MARL 不把非线性意见动力学当作观测滤波器、动作平滑器或普通循环网络，而是用它对多智能体协调问题进行**分岔提升（bifurcation lifting）**。

多智能体冲突通常具有多个互相分离的协调解。例如，对冲突车辆对 \(e=(i,j)\)：

\[
\mathcal M_e^+ : i\text{ 先行、}j\text{ 让行},
\qquad
\mathcal M_e^- : j\text{ 先行、}i\text{ 让行}.
\]

如果直接引入离散协调变量：

\[
\sigma_e\in\{-1,+1\},
\]

原问题会成为难以通过连续策略梯度求解的混合最优控制问题。BF-MARL 引入连续分岔状态 \(z_e\)：

\[
z_e=0\leftrightarrow\text{未决},\qquad
z_e>0\leftrightarrow\mathcal M_e^+,\qquad
z_e<0\leftrightarrow\mathcal M_e^-.
\]

完整分工为：

\[
\boxed{
\begin{aligned}
\text{冲突物理量} &\rightarrow \text{控制分岔临界性};\\
\text{MARL} &\rightarrow \text{学习最小分岔控制};\\
\text{NOD} &\rightarrow \text{生成、保持和切换协调分支};\\
\text{Actor} &\rightarrow \text{学习各分支下的连续控制};\\
\text{Critic} &\rightarrow \text{评估增广状态的长期任务价值}.
\end{aligned}}
\]

方法的核心不是“RL 学习意见”，而是：

\[
\boxed{\text{RL 求解一个包含低维分岔协调子系统的增广多智能体最优控制问题。}}
\]

---

## 2. 设计目标与非目标

### 2.1 设计目标

1. 在弱冲突下保持中性，不提前制造不必要的优先关系；
2. 在冲突接近临界点时，以小控制输入选择协调分支；
3. 分支形成后依靠双稳态保持承诺，避免高频抢行—让行切换；
4. 冲突解除后自动恢复中性；
5. 保持分散执行和参数共享；
6. 允许使用任意 PPO、MAPPO、IPPO 或其他 on-policy Actor–Critic 实现；
7. 使分岔状态成为增广系统的真实状态，而不是仅用于可视化的辅助变量；
8. 允许后接独立的 CBF、ORCA、MPC 或规则安全层，但不将安全层包装成主要创新。

### 2.2 非目标

BF-MARL 第一版不尝试：

- 学习完整物理动力学模型；
- 用反事实 Critic 构造意见标签；
- 人工指定“某种状态必须让行”的监督标签；
- 同时学习全部 NOD 参数；
- 将意见固定映射为纵向速度残差；
- 用普通 GRU 替代分岔状态；
- 声称仅凭意见动力学即可提供碰撞安全保证。

---

## 3. 增广多智能体最优控制问题

考虑 \(N\) 个智能体。物理状态和动作满足：

\[
x^{t+1}=f(x^t,a^t),
\qquad
a^t=(a_1^t,\ldots,a_N^t).
\]

智能体 \(i\) 只能获得局部观测：

\[
o_i^t=O_i(x^t).
\]

当前局部冲突图为：

\[
\mathcal G_t=(\mathcal V,\mathcal E_t).
\]

对每条冲突边 \(e\in\mathcal E_t\)，维护分岔状态 \(z_e\)。增广状态为：

\[
\widetilde x^t=(x^t,\mathbf z^t).
\]

策略同时产生物理动作分布和低维分岔控制：

\[
(a_i^t,b_i^t)\sim\pi_{\theta,i}(\cdot\mid o_i^t,\mathbf z_i^t),
\]

或者将 \(b_i^t\) 视为 Actor 内部的确定性可微输出，仅对物理动作计算策略 log-prob。

推荐优化目标为：

\[
\min_\theta
\mathbb E_{\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t
\left(
\ell_{\mathrm{task}}(x^t,a^t)
+\lambda_b\|\mathbf b^t\|_2^2
\right)
\right].
\]

其中 \(\lambda_b\|\mathbf b\|^2\) 是分岔控制能量，而不是意见幅值惩罚。稳定分支上的非零 \(z\) 是方法希望保留的协调承诺，不应被常规 \(z^2\) 正则压回零点。

---

## 4. 框架无关的数据接口

### 4.1 基本符号

| 符号 | 含义 |
|---|---|
| \(B\) | 并行环境或序列 batch 数 |
| \(N\) | 智能体数量 |
| \(K\) | 每个智能体最多保留的局部冲突邻居数 |
| \(F_o\) | 单智能体观测维度 |
| \(F_p\) | 成对交互特征维度 |
| \(D_a\) | 单智能体动作维度 |

### 4.2 环境/适配层输出

每个物理步至少产生：

```text
local_observation      float [B,N,Fo]
pair_features          float [B,N,K,Fp]
neighbor_ids           long  [B,N,K]
candidate_mask         bool  [B,N,K]
conflict_intensity     float [B,N,K] in [0,1]
confidence             float [B,N,K] in [0,1]
agent_reset_mask       bool  [B,N]
environment_done       bool  [B]
```

必须区分：

- `candidate_mask`：邻居是否真实存在且可观测；
- `conflict_intensity`：候选关系距离协调分岔临界点有多近；
- `active_conflict`：仅用于统计，不应用来决定是否创建意见状态。

如果只在强冲突发生后才创建状态，系统将失去临界点附近的提前选支能力。

### 4.3 推荐车辆对特征

一般连续运动任务可采用：

\[
\chi_{ij}=
[r_{ij},v_{ij},v_i,v_j,
\sin\Delta\psi,\cos\Delta\psi,
t_{\mathrm{CPA}},d_{\mathrm{CPA}},
\text{route/context features}].
\]

要求：

1. 全部特征在局部坐标系表达；
2. 有明确的交换变换 \(\mathcal S\chi_{ij}=\chi_{ji}\)；
3. 不输入未来真值轨迹、未来动作和中央 Critic 状态；
4. 全局 ID 只用于状态关联，不作为网络特征；
5. 数值归一化范围尽量落在 \([-1,1]\) 或 \([0,1]\)。

---

## 5. 连续冲突临界性

### 5.1 冲突强度

冲突强度 \(\rho_{ij}\in[0,1]\) 应当是连续、可解释、单调的物理量。一个通用例子是：

\[
\rho_{ij}
=
m_{ij}
\exp\left(
-\frac{t_{\mathrm{CPA}}}{\tau_t}
-\frac{d_{\mathrm{CPA}}}{\tau_d}
\right).
\]

如果具有路径冲突点，也可以基于预计到达时间、路径占用重叠或碰撞概率计算。第一版建议使用固定解析形式；若后续学习 \(\rho\)，需要另外验证单调性和标定误差。

### 5.2 临界参数

将冲突强度映射为 pitchfork normal form 的临界参数：

\[
\boxed{
\mu_{ij}
=
\mu_{\min}
+(\mu_{\max}-\mu_{\min})\rho_{ij},
}
\]

其中：

\[
\mu_{\min}<0<\mu_{\max}.
\]

临界冲突强度为：

\[
\rho_c
=
\frac{-\mu_{\min}}
{\mu_{\max}-\mu_{\min}}.
\]

解释为：

\[
\begin{cases}
\rho<\rho_c,\ \mu<0:& \text{中性单稳态};\\
\rho>\rho_c,\ \mu>0:& \text{两个稳定协调分支}.
\end{cases}
\]

第一版不让 MARL 学习 \(\mu_{\min},\mu_{\max},\rho_c\)。这些量决定方法的理论相图，应该作为可审计超参数或通过独立标定确定。

---

## 6. 最小分岔控制网络

### 6.1 反对称控制

使用共享成对评分器：

\[
s_{ij}=G_{\phi}(\chi_{ij}),
\qquad
s_{ji}=G_{\phi}(\mathcal S\chi_{ij}),
\]

并构造：

\[
\ell_{ij}=s_{ij}-s_{ji}.
\]

因此：

\[
\ell_{ji}=-\ell_{ij}.
\]

推荐网络：

```text
PairScorer
Fp → 128 → 128 → 1
activation: Tanh / SiLU
final layer: near-zero initialization
```

### 6.2 临界窗口控制

控制输入不应随冲突紧迫度持续增大，而应集中在最容易选支的临界窗口：

\[
g_{\mathrm{crit}}(\rho)
=
\exp\left[
-\frac{(\rho-\rho_c)^2}{2\sigma_c^2}
\right].
\]

最终控制为：

\[
\boxed{
b_{ij}
=
b_{\max}
m_{ij}c_{ij}
g_{\mathrm{crit}}(\rho_{ij})
\tanh\left(\frac{\ell_{ij}}{T_b}\right).
}
\]

该式保证：

\[
b_{ji}=-b_{ij},
\qquad
|b_{ij}|\le b_{\max}.
\]

其控制含义是：

1. 远离冲突时不干预；
2. 临界点附近用很小的长期价值驱动输入选择分支；
3. 分支形成后减小输入，让吸引子负责保持承诺；
4. 训练目标通过 \(b^2\) 惩罚促使策略采用最小必要干预。

### 6.3 精确对称状态

若 \(b=0,z=0,\mu>0\)，数学上 \(z=0\) 虽不稳定，但有限精度仿真可能仍停在零点。需要选择一种显式破缺机制：

1. 在无向边首次创建时采样极小的零均值反对称扰动；
2. 在临界窗口为 \(b\) 增加有界随机探索；
3. 使用可共享的一次性随机 edge token；
4. 若系统允许一标量通信，可协商随机种子。

扰动只能用于打破完全对称，不应成为固定车辆 ID 优先规则。评估时应报告两个分支被选中的比例，以检查公平性。

---

## 7. 近端分岔动力学层

### 7.1 连续形式

对每条边使用：

\[
\dot z
=
\mu z-z^3+b.
\]

对应势能：

\[
\Phi(z;\mu,b)
=
\frac14z^4
-\frac12\mu z^2
-bz.
\]

动力学满足：

\[
\dot z=-\frac{\partial\Phi}{\partial z}.
\]

### 7.2 离散近端形式

推荐用隐式近端更新：

\[
\boxed{
z^{t+1}
=
\arg\min_z
\left[
\frac{(z-z^t)^2}{2\eta}
+\Phi(z;\mu^t,b^t)
\right].
}
\]

一阶最优条件为：

\[
F(z)
=
\frac{z-z^t}{\eta}
+z^3-\mu^t z-b^t
=0.
\]

使用固定次数 Newton 迭代：

\[
z\leftarrow
z-
\frac{F(z)}
{1/\eta+3z^2-\mu^t}.
\]

推荐：

- 初值：\(z^{(0)}=z^t\)；
- Newton 次数：4–8；
- 使用 float32 时给分母加极小数值保护；
- 禁止对输出直接 `clamp`，除非仅作为最后的异常保护；
- 保持求解过程可微。

若：

\[
\eta\mu_{\max}<1,
\]

则单步近端目标严格凸，解唯一。冻结 \((\mu,b)\) 时还有：

\[
\Phi(z^{t+1})
+\frac{1}{2\eta}|z^{t+1}-z^t|^2
\le\Phi(z^t).
\]

### 7.3 分岔与迟滞

当 \(b=0\)：

\[
\begin{cases}
\mu<0:& z^\star=0\text{ 唯一稳定};\\
\mu>0:& z^\star=\pm\sqrt\mu\text{ 稳定，}z=0\text{ 不稳定}.
\end{cases}
\]

当 \(\mu>0\)，分支折叠阈值为：

\[
\boxed{
|b_{\mathrm{fold}}|
=
\frac{2\mu^{3/2}}{3\sqrt3}.
}
\]

低于该阈值的小扰动只能改变当前平衡位置，不能消除当前分支。因此迟滞裕度可以被计算和实验验证，而不是仅用“动作更平滑”描述。

---

## 8. 边状态与对称性约束

### 8.1 一个无向边只维护一个模态状态

对车辆对 \(\{i,j\}\)，只维护一个无向边状态 \(z_{\{i,j\}}\)，两辆车读取相反方向：

\[
z_{ij}=z_{\{i,j\}},
\qquad
z_{ji}=-z_{\{i,j\}}.
\]

等价地，稠密状态满足：

\[
\mathbf Z=-\mathbf Z^\top,
\qquad
\operatorname{diag}(\mathbf Z)=0.
\]

这比独立维护 \(z_{ij}\) 和 \(z_{ji}\) 更重要，因为互补分支输入是参数共享 Actor 突破对称动作子空间的基础。

### 8.2 生命周期

每个物理步按照以下规则更新：

1. 新出现的无向边从零或极小反对称扰动初始化；
2. 可见候选边使用当前 \((\mu,b)\) 做 Prox 更新；
3. 不可见但尚未重置的边使用 \(\mu_{\min},b=0\) 更新并回到中性；
4. 任一端智能体重置时清除整条边；
5. 环境结束时清空所有边；
6. 每个物理步只允许提交一次状态。

当 \(N\) 较小，可直接维护 \([B,N,N]\) 稠密反对称矩阵；大规模系统使用哈希边表或稀疏图张量。

### 8.3 可选的无环扩展

如果执行时允许交换一个标量节点势，可令：

\[
p_i=P_\phi(o_i),
\qquad
\mathbf b=B^\top\mathbf p.
\]

完成分支选择后，边方向与 \(p_i-p_j\) 一致，因而不会形成静态有向优先循环。该扩展需要明确通信或共享可观测性假设；无通信版本不应悄悄使用中央节点势。

---

## 9. 分支条件 Actor

### 9.1 原则

Actor 不使用人工公式把正意见映射为加速、负意见映射为减速。它应学习：

\[
\text{在当前协调分支和几何状态下，什么连续动作最优。}
\]

Actor 可以从零训练，也可以由已有 Actor warm-start。推荐拆成四个部分：

1. Ego/任务编码器；
2. 分支条件成对编码器；
3. 多邻居聚合器；
4. 动作分布头。

### 9.2 Ego编码器

通用默认结构：

```text
local observation [Fo]
 → Linear(Fo,256) + Tanh
 → Linear(256,256) + Tanh
 → h_ego [256]
```

### 9.3 分支变量

使用有界意见：

\[
q_{ij}=\tanh(z_{ij}/z_0).
\]

为强制意见真正参与模式选择，推荐使用分支门控的双专家结构：

\[
w_{ij}^{+}=\frac{1+q_{ij}}{2},
\qquad
w_{ij}^{-}=\frac{1-q_{ij}}{2}.
\]

两个分支专家分别产生：

\[
e_{ij}^{+}=E_{+}(\chi_{ij},\rho_{ij}),
\qquad
e_{ij}^{-}=E_{-}(\chi_{ij},\rho_{ij}).
\]

分支条件表示为：

\[
\boxed{
e_{ij}^{\mathrm{BF}}
=
w_{ij}^{+}e_{ij}^{+}
+w_{ij}^{-}e_{ij}^{-}.
}
\]

默认维度：

```text
BranchExpert+: (Fp+1) → 64 → 128
BranchExpert-: (Fp+1) → 64 → 128
```

专家名称只代表 \(+/-\) 分支，不预先规定具体动作。MARL 自主学习每个分支对应的连续行为。

如果希望更少参数，可共享底层 pair encoder，只保留两个小型 branch adapter。

### 9.4 多邻居聚合

推荐 masked attention：

\[
\alpha_{ij}
=
\operatorname{softmax}_{j:m_{ij}=1}
\left(
w_\alpha^\top e_{ij}^{\mathrm{BF}}
+\log(c_{ij}+\varepsilon)
\right),
\]

\[
c_i=\sum_j\alpha_{ij}e_{ij}^{\mathrm{BF}}.
\]

若没有候选邻居，则严格令：

\[
c_i=0.
\]

不要使用未经归一化的求和，否则邻居数量会无意放大控制幅值。

### 9.5 隐空间融合

推荐零初始化、受限的隐藏层 Adapter：

\[
\Delta h_i
=
\lambda_h\tanh(W_c c_i),
\]

\[
h_i^{\mathrm{BF}}
=
h_i^{\mathrm{ego}}+\Delta h_i.
\]

再输出动作分布：

\[
(\mathrm{loc}_i,\mathrm{scale}_i)
=H_a(h_i^{\mathrm{BF}}),
\]

\[
a_i\sim\operatorname{TanhNormal}
(\mathrm{loc}_i,\mathrm{scale}_i).
\]

若 \(W_c\) 严格零初始化，则网络初始时精确退化为不使用分岔上下文的 Base Actor。这有利于 warm-start、数值等价测试和消融。

### 9.6 不推荐的接口

正式主方法不建议：

- \(a=a_{\mathrm{base}}+c z\) 的固定动作残差；
- 只修改速度、不允许意见影响转向；
- 将 \(b\) 直接拼入动作头；
- 将 Critic 的 \(\Delta Q\) 当成监督标签；
- 让 Actor 同时任意学习 \(\mu\)、\(b\) 和 NOD 非线性系数。

---

## 10. 中央 Critic

### 10.1 为什么必须读取意见状态

BF-MARL 的 Markov 状态是 \((x,z)\)。相同物理状态下，“已经承诺先行”和“已经承诺让行”的未来回报可能不同。如果 Critic 不读取 \(z\)，价值估计会产生状态混叠。

每车构造：

\[
o_{i}^{V}
=
[o_i,
\operatorname{sg}(z_{i,:}),
\rho_{i,:},
m_{i,:}].
\]

然后使用任意中央图 Critic 或集中 MLP：

```text
per-agent critic feature
 → centralized MLP / GNN / attention critic
 → V_i or shared team value
```

`stop_gradient(z)` 表示价值损失不得通过 Critic 输入更新分岔控制网络；Actor 梯度仍通过策略分布中的 \(z\) 正常传播。

### 10.2 Critic选择

| 场景 | 推荐 Critic |
|---|---|
| 固定少量智能体 | 拼接式中央 MLP |
| 数量变化但较小 | 图注意力 Critic |
| 大规模同质群体 | 局部图 Critic或值分解 |
| 团队共享回报 | shared team value 或每车同值输出 |

Critic结构不是方法创新中心；关键是其输入必须包含增广协调状态。

---

## 11. 单步因果顺序

每个物理步只能使用以下顺序：

```text
1. 读取 local observation 和当前物理状态
2. 构造候选冲突图与 pair features
3. 根据 reset mask 清理相关无向边状态
4. 计算 conflict intensity rho 和 criticality mu
5. BifurcationControlNet 计算反对称 b
6. ProximalBifurcationLayer: z_prev → z_next
7. 使用 z_next 构造 branch-conditioned pair embeddings
8. 聚合 coordination context
9. Actor 输出 loc/scale，采样 nominal action 和 log_prob
10. 可选安全层将 nominal action 投影为 executed action
11. env.step(executed action)
12. 将 z_next detach 后提交到 rollout edge state
```

Actor必须使用当前刚计算的 \(z_{t+1}\)。如果使用旧 \(z_t\)，当前 \(b_t\) 无法影响当前动作，单步策略梯度会变弱。

安全层若存在，应被视为环境闭环的一部分。必须同时记录 nominal action、executed action 和 intervention mask，防止把安全层收益错误归因于 BF-MARL。

---

## 12. 序列 PPO

### 12.1 为什么不能只做单步 PPO

BF-MARL 的核心收益来自：

\[
b_t
\rightarrow z_{t+1}
\rightarrow z_{t+2}
\rightarrow\cdots
\rightarrow a_{t+k}
\rightarrow R.
\]

若每个物理步都 detach \(z\)，策略只能学习 \(b_t\) 对当前动作的即时影响，无法学习“临界点短暂干预—后续零输入保持”的控制策略。

因此正式训练必须保留连续 chunk，并在 chunk 内重新展开分岔层。

### 12.2 Rollout存储

每个时间步至少保存：

```text
observation
pair_features
neighbor_ids
candidate_mask
conflict_intensity
confidence
agent_reset_mask
z_prev / edge-state snapshot
nominal_action
executed_action (if shield exists)
old_log_prob
reward
done
value estimate
```

所有采样数据 detach。每个 chunk 额外保存：

```text
z_init
edge-validity init
random symmetry-breaking token init, if used
```

### 12.3 PPO重算

在 minibatch 中：

1. chunk 维并行；
2. 时间维循环；
3. 从 `z_init.detach()` 开始；
4. 重算每个时间步的 \(b,\mu,z,\mathrm{loc},\mathrm{scale}\)；
5. 使用存储动作计算新 log-prob；
6. chunk内部保留完整计算图；
7. reset边界立即清除对应意见状态。

推荐 chunk 时间长度覆盖一次主要冲突形成过程。若物理步长为 \(\Delta t\)，冲突形成时间尺度为 \(T_c\)，建议：

\[
L_{\mathrm{chunk}}\Delta t\gtrsim T_c.
\]

### 12.4 策略损失

\[
r_t(\theta)
=
\frac{pi_\theta(a_t\mid o_t,z_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid o_t,z_t)},
\]

\[
\mathcal L_{\mathrm{clip}}
=
-\mathbb E
\left[
\min
\left(
r_t\widehat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\widehat A_t
\right)
\right].
\]

总 Actor 损失建议：

\[
\boxed{
\mathcal L_{\mathrm{actor}}
=
\mathcal L_{\mathrm{clip}}
-\lambda_H\mathcal H
+\lambda_b\mathbb E[m_{ij}b_{ij}^2]
+\lambda_{\mathrm{sat}}\mathcal L_{\mathrm{sat}}.
}
\]

其中 \(\mathcal L_{\mathrm{sat}}\) 只用于抑制控制输入长期饱和。不要加入通行标签、\(\Delta Q\) 教师或意见符号监督。

总训练损失为：

\[
\mathcal L
=
\mathcal L_{\mathrm{actor}}
+c_V\mathcal L_V.
\]

Actor和Critic最好使用独立参数组与梯度裁剪。

---

## 13. 框架无关伪代码

### 13.1 Rollout

```python
def policy_step(obs, physical_state, edge_state):
    pair = build_local_conflict_graph(physical_state)
    edge_state.clear_resets(pair.agent_reset_mask)

    rho = compute_conflict_intensity(pair)
    mu = criticality_map(rho)
    b = bifurcation_control_net(
        pair.features,
        swapped(pair.features),
        rho,
        pair.confidence,
        pair.candidate_mask,
    )

    z_prev = edge_state.read_dense()
    z_next = proximal_bifurcation_step(z_prev, mu, b)
    z_next = enforce_skew_symmetry(z_next)

    ego_hidden = ego_encoder(obs)
    branch_context = branch_conditioned_context(
        pair.features, z_next, rho, pair.candidate_mask
    )
    fused_hidden = ego_hidden + bounded_adapter(branch_context)
    loc, scale = action_head(fused_hidden)

    nominal_action, log_prob = sample_tanh_normal(loc, scale)
    executed_action = optional_safety_layer(nominal_action, physical_state)

    edge_state.commit(stop_gradient(z_next))
    return nominal_action, executed_action, log_prob, diagnostics
```

### 13.2 Sequence PPO重算

```python
def recompute_chunk(chunk):
    z = stop_gradient(chunk.z_init)
    new_log_probs = []
    control_energy = 0.0

    for t in range(chunk.length):
        z = apply_resets(z, chunk.reset_mask[t])
        rho_t = chunk.conflict_intensity[t]
        mu_t = criticality_map(rho_t)
        b_t = bifurcation_control_net(...)
        z = proximal_bifurcation_step(z, mu_t, b_t)

        dist_t = actor_distribution(
            chunk.observation[t],
            chunk.pair_features[t],
            z,
            rho_t,
        )
        new_log_probs.append(dist_t.log_prob(chunk.nominal_action[t]))
        control_energy += masked_mean(b_t ** 2)

    return stack(new_log_probs), control_energy
```

该伪代码可直接映射到 PyTorch、JAX、TensorFlow 或其他自动微分框架。

---

## 14. 推荐默认配置

以下仅作为第一轮稳定实现的起点，不能替代场景标定。

| 模块 | 推荐初值 |
|---|---|
| Pair scorer | `Fp → 128 → 128 → 1` |
| Ego encoder | `Fo → 256 → 256` |
| Branch expert | `(Fp+1) → 64 → 128`，两个分支 |
| Coordination adapter | `128 → 256`，输出层零初始化 |
| Actor head | `256 → 2*Da` |
| Critic | 2层256或图注意力网络 |
| \(\mu_{\min}\) | `-0.5` |
| \(\mu_{\max}\) | `1.0` |
| \(\eta\) | 满足 `eta * mu_max < 1`，可从 `0.1` 开始 |
| \(b_{\max}\) | 从 `0.1–0.3` 开始 |
| \(T_b\) | `1.0` |
| \(\sigma_c\) | `0.1–0.2` |
| Newton steps | `6` |
| PPO clip | `0.2` |
| GAE \(\gamma\) | `0.99` |
| GAE \(\lambda\) | `0.9–0.95` |
| Actor learning rate | `1e-4–3e-4` |
| Critic learning rate | `2e-4–5e-4` |
| Gradient norm | `0.5–1.0` |
| Chunk长度 | 覆盖一次主要冲突形成时间 |

标定顺序应为：

1. 先确定 \(\rho\) 的物理范围；
2. 再选择 \(\rho_c\)；
3. 再确定 \(\mu_{\min},\mu_{\max}\)；
4. 选择满足唯一近端解条件的 \(\eta\)；
5. 最后逐步增加 \(b_{\max}\) 和降低 \(\lambda_b\)。

---

## 15. 推荐训练阶段

### 阶段0：无分岔Base

训练或加载普通 MAPPO Actor–Critic，得到基本任务能力。该阶段也可直接省略，进入从零联合训练。

### 阶段1：零控制等价

令：

\[
b_{\max}=0
\]

或将 Coordination Adapter 输出层严格置零。验证：

- 动作分布与 Base 完全一致；
- 无冲突时 \(z\rightarrow0\)；
- 启用模块不会改变环境观测、奖励和 reset；
- rollout 与 PPO重算得到相同状态轨迹。

### 阶段2：冻结Base，仅训练分岔路径

短期训练 BifurcationControlNet、Branch Experts 和 Adapter，验证梯度路径与数值稳定性。该阶段只用于工程调试，不作为最终公平性能结果的必要训练预算。

### 阶段3：正式联合训练

联合训练：

- Ego Encoder；
- BifurcationControlNet；
- Branch Experts；
- Coordination Adapter；
- Actor Head；
- Central Critic。

固定：

- 临界性映射；
- Proximal Bifurcation Layer；
- 对称性投影；
- 可选安全层。

正式论文结果应使用与 Base 相同的总采样预算。从已训练 Base 开始的 warm-start 结果单独报告。

### 阶段4：课程学习，可选

如果从零训练不稳定，可按以下顺序增加难度：

1. 两智能体对称冲突；
2. 两智能体非对称冲突；
3. 三至四智能体稀疏冲突；
4. 密集动态冲突；
5. 感知噪声和未见密度。

课程学习只改变初始状态分布，不改变方法公式。

---

## 16. 必须实现的数值和单元测试

### 16.1 分岔层测试

1. \(\mu<0,b=0\) 时从任意小初值收敛到零；
2. \(\mu>0,b=0\) 时正负初值分别收敛到 \(\pm\sqrt\mu\)；
3. \(b\) 改变符号时分支方向按预期倾斜；
4. Newton残差 \(|F(z)|\) 小于设定阈值；
5. Prox能量不增；
6. 自动微分梯度通过有限差分检查；
7. \(\eta\mu_{\max}<1\) 配置检查失败时立即报错。

### 16.2 对称性测试

1. `swap(swap(pair_features)) == pair_features`；
2. \(b_{ji}=-b_{ij}\)；
3. \(z_{ji}=-z_{ij}\)；
4. 智能体标签交换后联合动作分布相应交换；
5. 精确对称状态下不会永久停留在 \(z=0\)；
6. 对称场景中两个分支长期选择频率不存在显著固定ID偏置。

### 16.3 状态生命周期测试

1. 新边初始化正确；
2. 消失边回归中性；
3. 单车 reset 同时清除入边和出边；
4. 环境 done 后状态完全清零；
5. 每个物理步仅提交一次；
6. 并行环境之间无状态串扰。

### 16.4 PPO一致性测试

1. rollout log-prob 与无参数更新时的重算 log-prob 一致；
2. rollout \(z\) 与 sequence replay \(z\) 一致；
3. Critic loss 不产生 Actor梯度；
4. Actor loss可到达 BifurcationControlNet；
5. chunk reset 边界不会跨episode传播梯度；
6. 所有存储动作和旧 log-prob 已 detach。

---

## 17. 诊断指标

### 17.1 任务指标

- 联合回报；
- 成功率/路线完成率；
- 碰撞率和最小间距；
- 平均通行时间、冲突解决时间；
- 停滞率、超时率和死锁率；
- 加速度、jerk、转向变化等舒适性指标。

### 17.2 分岔指标

- \(\rho,\mu,b,z\) 的时间曲线；
- 临界点到形成稳定分支的延迟；
- 分支选择正确率或长期价值一致率；
- 分支切换次数；
- 在 \(|b|<b_{\mathrm{fold}}\) 噪声下的误切换率；
- 冲突解除后的中性恢复时间；
- \(b\) 的均值、P95、最大值和饱和率；
- 单次冲突的分岔控制能量 \(\sum_t b_t^2\)；
- 临界窗口外的控制泄漏；
- 对称任务中的分支公平性。

### 17.3 策略使用指标

- \(\|\partial\mathrm{loc}/\partial z\|\) 或对应敏感度；
- 分支专家使用比例；
- 去掉 \(z\) 后的动作分布变化；
- nominal/executed action差异；
- 安全层干预率；
- Critic在相同物理状态、不同 \(z\) 下的值差异。

全时域均值可能掩盖稀疏冲突，应同时报告冲突窗口指标。

---

## 18. 必要消融

至少包含：

| 消融 | 回答的问题 |
|---|---|
| Base-MARL | 基础性能 |
| Base-MARL + GRU | 提升是否仅来自循环记忆 |
| 直接 \(b\rightarrow\) Actor | 是否真的需要分岔动力学 |
| 线性滤波状态 | 分岔是否优于低通滤波 |
| 显式 Euler NOD | Prox离散化是否改善稳定性 |
| 无 critical gate | 临界窗口控制是否必要 |
| 无 \(b^2\) 控制代价 | 最小干预是否成立 |
| 独立 \(z_{ij},z_{ji}\) | 反对称边状态是否必要 |
| 固定动作残差 | 分支条件 Actor 是否更有表达力 |
| Critic不读 \(z\) | 增广价值建模是否必要 |
| 单步 PPO | 长期分岔控制是否依赖序列梯度 |
| 完整 BF-MARL | 主方法 |

如果存在安全层，应让所有可比较 MARL 方法使用同一安全层，并额外报告无安全层结果或安全层干预率。

---

## 19. 理论结果建议

论文应优先建立以下结果，而不是只证明有界性。

### 19.1 模态—平衡分支对应

证明 \(\mu<0\) 时只有中性稳定平衡，\(\mu>0\) 时产生两个稳定协调分支，从而给出离散协调模态的连续实现。

### 19.2 近端离散稳定性

证明 \(\eta\mu_{\max}<1\) 时单步近端映射单值，且冻结输入时满足能量下降。

### 19.3 迟滞抗扰裕度

由：

\[
|b_{\mathrm{fold}}|
=\frac{2\mu^{3/2}}{3\sqrt3}
\]

建立输入噪声到分支保持的显式裕度。

### 19.4 对称策略可表示性

在交换对称观测与参数共享策略下，未增广的确定性策略受限于对称动作子空间；加入满足 \(z_{ij}=-z_{ji}\) 的分岔状态后，共享 Actor可以表示一对互补、交换对称的非对称联合动作。

### 19.5 慢变分支跟踪

若 \(\rho_t,b_t\) 相对意见子系统慢变并远离折叠点，证明 \(z_t\) 跟踪稳定平衡分支，误差为时间尺度比的高阶小量。

### 19.6 可选的无环性

若采用节点势控制 \(b=B^\top p\)，证明稳定边方向对应势函数下降方向，静态优先图无有向循环。

---

## 20. 常见失败模式

### 20.1 将 \(b\) 乘以普通 urgency

这会让控制在最危险时最强，把 NOD重新变成紧急动作修正器。应使用临界窗口 gate。

### 20.2 Actor直接读取 \(b\)

网络可能绕过 NOD。正式新增协调路径应从 \(z\) 进入 Actor。

### 20.3 两个方向独立维护意见

会出现 \(z_{ij}>0,z_{ji}>0\) 等相互矛盾状态，破坏互补分支语义。

### 20.4 Critic忽略 \(z\)

会产生增广状态混叠，使 GAE优势估计噪声增大。

### 20.5 单步截断全部时间梯度

网络只能学习瞬时控制，无法学习最小分岔干预。

### 20.6 同时学习所有动力学参数

临界点、势阱深度和控制幅值同时漂移，理论相图和训练语义都会失去稳定参照。

### 20.7 惩罚意见幅值

\(z^2\) 正则会抵消稳定承诺。应惩罚 \(b^2\)，而不是稳定分支本身。

### 20.8 忽略精确对称零点

数值系统可能永久停在不稳定的零平衡，必须有明确的无偏破缺机制。

### 20.9 安全层完全接管动作

如果 nominal/executed action差异长期很大，性能主要来自安全层而非学习策略。必须单独审计。

---

## 21. 可移植模块接口

无论底层使用何种代码库，只需实现以下接口：

```text
ConflictGraphAdapter
    physical_state/local_obs
      → pair_features, ids, masks, rho, confidence

BifurcationControlNet
    pair_features, swapped_features, rho, confidence, mask
      → b

ProximalBifurcationLayer
    z_prev, mu, b, reset_mask
      → z_next

EdgeStateStore
    read / reset / commit / snapshot

BifurcationConditionedActor
    local_obs, pair_features, z_next, rho, mask
      → action_distribution

AugmentedCentralCritic
    global/local observations, stopgrad(z_prev), rho, mask
      → value

SequenceRolloutBuffer
    contiguous chunks + z_init + reset boundaries

SequencePolicyOptimizer
    recompute temporal policy + PPO/other on-policy objective
```

环境适配层之外的模块不应读取具体仿真器对象。动作边界、状态维数、车辆数量和候选数全部通过配置或张量 shape 注入。

---

## 22. 验收标准

一个实现只有同时满足以下条件，才能称为 BF-MARL：

1. 冲突强度跨过临界值时确实发生单稳态到双稳态转换；
2. MARL学习的是有界分岔控制，而不是直接意见标签；
3. 分岔控制主要集中于临界窗口；
4. Actor根据分岔后的 \(z\) 学习分支条件连续控制；
5. 不使用固定速度残差作为正式接口；
6. 边状态具有方向反对称性；
7. Critic读取停止梯度的增广意见状态；
8. 正式训练在连续 chunk 内保留时间梯度；
9. Prox更新满足数值残差和能量下降测试；
10. 无冲突时意见自动回中性；
11. 分支形成后的小噪声不造成高频切换；
12. 与 Base、GRU、滤波、直接控制和显式Euler版本完成同预算消融。

---

## 23. 最终方法摘要

BF-MARL 将多智能体协调建模为受长期任务回报驱动的最小能量分岔控制问题。物理冲突强度决定系统何时从中性单稳态进入双稳态协调区；共享的分岔控制网络在临界窗口施加有界、反对称的小输入，选择协调分支；近端非线性意见动力学将该选择转化为具有迟滞和恢复性的动态承诺；分支条件 Actor学习每个协调模态下的连续控制；中央 Critic则对包含意见状态的增广系统进行价值评估。

完整闭环为：

\[
\boxed{
\text{局部物理交互}
\rightarrow
\text{临界性}\ \mu
\rightarrow
\text{最小分岔控制}\ b
\rightarrow
\text{Prox-NOD分支}\ z
\rightarrow
\text{分支条件动作策略}
\rightarrow
\text{长期联合回报}.
}
\]

最简洁的分工是：

\[
\boxed{
\text{MARL决定何时、向哪个方向施加最小干预；}
\quad
\text{NOD决定协调模式如何生成、保持、切换和消失。}
}
\]

---

## 24. 参考实现与理论风格来源

- DGPPO: <https://arxiv.org/abs/2502.03640>
- Distributed Epigraph Form MARL: <https://arxiv.org/abs/2504.15425>
- Nonlinear Opinion Dynamics with Tunable Sensitivity: <https://arxiv.org/abs/2009.04332>
- Game-Induced Nonlinear Opinion Dynamics: <https://arxiv.org/abs/2304.02687>
- 工程参考项目：<https://github.com/Allvey/sigmarl-nod>

