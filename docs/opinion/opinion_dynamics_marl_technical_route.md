# 意见动力学 + MARL：核心思想与技术路线

> 文档角色：当前方法的最高级理论真源  
> 代码底座：SigmaRL 1.2.0（tag commit
> `5fe715bdfba4ff3e33d901d69dfa220f1222c060`）  
> 方法关系：SigmaRL 提供环境与 Base-MAPPO；AVOCADO 提供意见动力学启发；TSC
> 仅作为外部实验基线  
> 对齐日期：2026-08-23

本文件描述独立的 Opinion Dynamics + MARL 方法，不是 TSC 的扩展。文中的“道路
拓扑”仅指 SigmaRL 1.2.0 环境已有的地图、参考路径和可观测几何关系，不等价于旧
TSC 的 `TopologyLearner`、priority graph 或 leader/follower 结构。

与当前 SigmaRL 1.2.0 实现对齐的第一版还固定：每车只处理原始局部观测中的两个
最近邻，使用当前相对位置/速度的短时常速外推判断潜在冲突。不使用未来真值
轨迹、真实未来动作或旧 TSC 的 action predictor。

## 1. 方法目标

本方案面向无显式通信条件下的多智能体车辆冲突协调，将多车交互建模为一个**由长期任务回报驱动的非线性意见形成过程**。

核心链路为：

\[
\boxed{
\text{局部物理交互}
\rightarrow
\text{瞬时意见证据 }b
\rightarrow
\text{动态意见状态 }z
\rightarrow
\text{动作分布}
}
\]

同时，任务结果通过 MAPPO 反向训练意见证据网络：

\[
\boxed{
\text{长期任务回报}
\rightarrow
\text{策略优势}
\rightarrow
B_{\phi_b}
\rightarrow
b
}
\]

由此形成双向耦合：

- 意见动力学在线决定证据如何积累、保持、切换和消退；
- MARL 离线学习什么物理情境应产生什么方向和强度的证据；
- 任务回报评价这种意见形成过程是否改善联合驾驶行为。

需要特别区分：

\[
\boxed{
b\text{ 在每个时刻被即时计算，但证据网络参数不会在执行时即时更新。}
}
\]

执行阶段参数 \(\phi_b\) 固定；只有在训练阶段，才通过 PPO 批量、低学习率地更新 \(\phi_b\)。

---

## 2. 变量语义与模块分工

对于车辆 \(i\) 与冲突车辆 \(j\)，车辆 \(i\) 维护一个有向意见状态：

\[
z_{ij}^t\in\mathbb R.
\]

其语义为：

\[
z_{ij}^t>0:
\quad i\text{ 倾向于相对 }j\text{ 继续通行},
\]

\[
z_{ij}^t<0:
\quad i\text{ 倾向于相对 }j\text{ 让行},
\]

\[
z_{ij}^t\approx0:
\quad \text{当前尚未形成明确意见}.
\]

系统中的三个核心部分具有不同职责：

| 模块 | 作用 |
|---|---|
| 意见证据 \(b_{ij}^t\) | 表示当前时刻局部物理交互提供的瞬时证据，不承担记忆 |
| 意见状态 \(z_{ij}^t\) | 对历史证据进行积累、衰减和自强化，形成连续协调意见 |
| MAPPO | 根据长期联合回报学习局部物理情境到 \(b_{ij}^t\) 的映射 |

因此：

\[
\boxed{
b=\text{瞬时证据},
\qquad
z=\text{具有时间连续性的协调承诺}.
}
\]

---

## 3. 总体运行流程

系统采用集中训练、分散执行（CTDE）。训练阶段使用中央 Critic；执行阶段每辆车只使用局部观测、局部历史和本地意见状态，不需要交换意见。

每个时刻按照以下因果顺序运行：

\[
\boxed{
o_i^t
\rightarrow
\chi_{ij}^t
\rightarrow
b_{ij}^t
\rightarrow
z_{ij}^t
\rightarrow
q_{ij}^t
\rightarrow
\mu_i^t
\rightarrow
a_i^t
}
\]

具体步骤如下：

1. 车辆 \(i\) 获取当前局部观测 \(o_i^t\)；
2. 对当前两个可观测最近邻做短时运动学外推，识别冲突车辆 \(j\)；
3. 构造车辆对局部交互特征 \(\chi_{ij}^t\)；
4. 证据网络生成有界的瞬时证据 \(b_{ij}^t\)；
5. 固定意见动力学将 \(z_{ij}^{t-1}\) 更新为 \(z_{ij}^t\)；
6. 将意见转换为有限的动作残差；
7. Actor 输出动作分布并采样动作；
8. 联合驾驶结果形成任务回报；
9. MAPPO 使用任务优势更新 Actor 和证据网络。

---

## 4. 冲突关系与局部交互特征

在时刻 \(t\) 构造动态冲突图：

\[
\mathcal G_t=(\mathcal V_t,\mathcal E_t).
\]

若车辆 \(i\) 和 \(j\) 按当前相对位置与速度做短时常速外推后，在预测时间窗内的
最近间距小于冲突阈值，则建立边：

\[
(i,j)\in\mathcal E_t.
\]

车辆 \(i\) 只为当前相关的冲突车辆维护意见状态：

\[
\mathbf z_i^t=\{z_{ij}^t:j\in\mathcal N_i^t\}.
\]

新冲突边出现时令 \(z_{ij}=0\)；冲突解除后，意见衰减至零并删除对应状态。

局部特征 \(\chi_{ij}^t\) 可包括：

- ego 坐标系中的相对位置和相对速度；
- 双方速度和相对航向；
- 最近接近时间（time to closest approach）；
- 外推时间窗口内的最小间距；
- 当前感知距离与有效 mask；
- 只由上述当前物理量计算的冲突紧迫度与置信度。

道路规则或地图几何可以在后续版本作为显式、静态的物理特征，但不得被表述为
学习的 priority、leader 或 TSC topology。

邻车行为预测器不属于第一版。若以后作为独立扩展，可计算：

\[
\widehat y_{ij}^{t|t-1}
=
P_\psi(h_i^{t-1},y_{ij}^{t-1},a_i^{t-1,\mathrm{exec}}),
\]

\[
\varepsilon_{ij}^t
=
y_{ij}^t-\widehat y_{ij}^{t|t-1}.
\]

预测残差只能是证据网络的一类物理输入，不是意见动力学的核心组成部分；该扩展
不得复用 TSC action predictor。

证据网络禁止读取：

\[
z_{ij}^t,\qquad q_{ij}^t,
\]

以避免形成 \(z\rightarrow b\rightarrow z\) 的非物理内部正反馈。

---

## 5. 可学习的意见证据 \(b\)

MARL 当前只学习外部证据网络：

\[
b_{ij}^t=B_{\phi_b}(\chi_{ij}^t).
\]

为了赋予正负号清晰的相对语义，推荐采用共享相对评分结构：

\[
\ell_{ij}^t
=
G_{\phi_b}(\xi_i^t,\xi_j^t,e_{ij}^t)
-
G_{\phi_b}(\xi_j^t,\xi_i^t,e_{ij}^t).
\]

最终证据为：

\[
\boxed{
b_{ij}^t
=
b_{\max}
\rho_{ij}^t
c_{ij}^t
\tanh\left(\frac{\ell_{ij}^t}{T_b}\right)
}
\]

其中：

- \(\rho_{ij}^t\in[0,1]\)：物理冲突紧迫度；
- \(c_{ij}^t\in[0,1]\)：感知或预测置信度；
- \(b_{\max}\)：证据幅值上限；
- \(T_b\)：输出温度。

该结构保证：

\[
|b_{ij}^t|\leq b_{\max}\rho_{ij}^tc_{ij}^t.
\]

因此：

- 无有效冲突时 \(b\approx0\)；
- 信息不可靠时证据自动减弱；
- 网络不能通过无限放大 \(b\) 绕过意见动力学；
- \(b\) 只承担瞬时证据作用，不承担时间记忆。

---

## 6. 固定的非线性意见动力学

意见根据当前证据更新：

\[
\boxed{
z_{ij}^{t}
=
z_{ij}^{t-1}
+
\Delta t\,\eta_z
\left[
-\kappa_z z_{ij}^{t-1}
+
\rho_{ij}^t\nu_z
\tanh(\alpha_z z_{ij}^{t-1})
+
b_{ij}^t
\right]
}
\]

其中：

- \(\kappa_z\)：意见遗忘强度；
- \(\nu_z\)：冲突期间的意见自强化强度；
- \(\alpha_z\)：非线性敏感度；
- \(\eta_z\)：意见响应速度。

三项动力学含义分别为：

\[
-\kappa_z z:
\quad \text{遗忘和回归中性},
\]

\[
\rho\nu_z\tanh(\alpha_z z):
\quad \text{冲突期间保持和强化已形成的意见},
\]

\[
b:
\quad \text{当前物理交互提供的新证据}.
\]

当前阶段固定：

\[
\eta_z,\kappa_z,\nu_z,\alpha_z.
\]

MARL 不直接改变动力学参数，只学习 \(b\)。这样可以保留意见动力学的解释性，并降低策略和动力学同时变化造成的训练非平稳性。

当冲突解除时：

\[
\rho_{ij}^t=0,\qquad b_{ij}^t=0,
\]

意见按照：

\[
\dot z_{ij}=-\eta_z\kappa_z z_{ij}
\]

衰减回零。

---

## 7. 稳定性说明

虽然 \(b_{ij}^t\) 每个时刻都可能变化，但它只是有界的外部输入，而不是在线改变动力学参数。

由于：

\[
|b_{ij}^t|\leq b_{\max},
\qquad
|\tanh(\alpha_z z)|\leq1,
\]

意见状态具有最终有界性：

\[
\limsup_{t\rightarrow\infty}|z_{ij}(t)|
\leq
\frac{\nu_z+b_{\max}}{\kappa_z}.
\]

因此，瞬时证据变化不会使意见无限发散。

采用意见动力学的目的不是对 \(b\) 再做一次静态微调，而是将可能含噪的瞬时证据转换为连续、具有记忆的协调状态：

\[
\boxed{
b=\text{瞬时证据},
\qquad
z=\text{经过积累、衰减和自强化后的稳定意见}.
}
\]

如果直接让 \(b\) 控制动作，单帧噪声可能导致抢行和让行行为频繁切换；意见动力学为策略提供了时间连续性和承诺保持能力。

---

## 8. 意见与 Actor 的耦合

首先将意见压缩到有限区间：

\[
q_{ij}^t
=
\tanh\left(\frac{z_{ij}^t}{z_0}\right)
\in[-1,1].
\]

Actor 分为基础动作分支和意见残差分支。

基础分支只读取物理观测：

\[
\mu_{i,\mathrm{base}}^t
=
f_\theta(o_i^t).
\]

意见残差为：

\[
\Delta\mu_{i,\mathrm{op}}^t
=
c_{\mathrm{op}}
\sum_{j\in\mathcal N_i^t}
\bar\rho_{ij}^t
q_{ij}^t
\mathbf d_{ij}^t,
\]

其中：

\[
\bar\rho_{ij}^t
=
\frac{\rho_{ij}^t}
{\epsilon+\sum_{k\in\mathcal N_i^t}\rho_{ik}^t}.
\]

最终策略均值为：

\[
\boxed{
\mu_i^t
=
\mu_{i,\mathrm{base}}^t
+
\Delta\mu_{i,\mathrm{op}}^t
}
\]

\(\mathbf d_{ij}^t\) 是固定或强约束的通行相关动作方向：

- 对纵向加速度控制，可令 \(\mathbf d_{ij}=1\)；
- 对多维动作，只调节沿参考路径的前进分量；
- 不让意见直接控制与通行权无关的转向分量。

为了避免 Actor 绕过意见模块，应保持：

\[
c_{\mathrm{op}}>0,
\qquad
\|\mathbf d_{ij}^t\|=1,
\]

并禁止基础动作分支直接读取 \(z\)。

---

## 9. MAPPO 对意见证据的训练

所有车辆共享分散 Actor：

\[
\pi_{\theta,\phi_b}.
\]

训练阶段使用中央 Critic。为保持 SigmaRL 1.2.0 的网络结构和 checkpoint 边界，
第一版只读取原始联合观测：

\[
V_\omega(x_t).
\]

将 \(\operatorname{sg}[\mathbf z_t]\) 追加到 Critic 只是后续可选消融，不进入第一版
主路径。无论 Critic 是否读取 detached \(z\)，价值损失都不得更新证据网络。

证据网络接受的任务梯度路径为：

\[
\boxed{
\phi_b
\rightarrow
b_{ij}^t
\rightarrow
z_{ij}^t
\rightarrow
q_{ij}^t
\rightarrow
\mu_i^t
\rightarrow
\log\pi_i
\rightarrow
\widehat A_t
}
\]

Actor 目标为：

\[
\mathcal L_{\mathrm{actor}}
=
-\mathcal L_{\mathrm{PPO}}
-\lambda_H\mathcal H(\pi)
+\lambda_{\mathrm{neutral}}\mathcal L_{\mathrm{neutral}}
+\lambda_{\mathrm{mag}}\mathcal L_{\mathrm{mag}},
\]

其中：

\[
\mathcal L_{\mathrm{neutral}}
=
\mathbb E\left[(1-\rho_{ij}^t)(b_{ij}^t)^2\right],
\]

\[
\mathcal L_{\mathrm{mag}}
=
\mathbb E\left[(b_{ij}^t)^2\right].
\]

中央 Critic 使用独立价值损失：

\[
\mathcal L_V
=
\mathbb E\left[
\left(V_\omega-\widehat G_t\right)^2
\right].
\]

本方案不使用：

- 人工意见标签；
- \(b\) 与具体动作之间的监督损失；
- 反事实分支 Critic；
- \(\Delta Q\) 教师蒸馏。

---

## 10. 序列化 PPO 训练

由于意见是循环状态：

\[
b^t\rightarrow z^t\rightarrow z^{t+1}\rightarrow\cdots,
\]

训练样本必须保留连续时间片段，不能完全打乱为独立单步样本。

Rollout Buffer 除普通 MAPPO 数据外，还需保存：

- 每个训练片段起点的 \(z_{ij}\)；
- 冲突边及车辆对应关系；
- 有效边掩码；
- episode 终止掩码；
- 旧策略动作概率。

原始 SigmaRL Actor 是无状态前馈 MLP，因此第一版没有其他 Actor 循环状态需要保存。

更新时：

1. 从保存的初始意见状态开始；
2. 在连续片段内部重新计算 \(b^t\)、\(z^t\) 和策略分布；
3. 片段起点的意见状态停止梯度；
4. 片段内部采用截断时间反向传播；
5. 冲突边消失或 episode 结束时正确重置状态。

环境状态转移不需要可微，梯度只需要通过：

\[
B_{\phi_b}
\rightarrow
\text{OpinionDynamics}
\rightarrow
\text{Actor}
\]

传播。

---

## 11. 推荐训练流程

### 阶段一：基础 MAPPO 预训练

令：

\[
b=0,\qquad z=0,
\]

先训练基础 Actor 和中央 Critic，使策略具备基本驾驶、避碰和通行能力。

### 阶段二：学习意见证据

启用意见模块：

- 固定全部意见动力学参数；
- 固定意见—动作残差方向；
- 第一版固定基础 Actor；缓慢更新 Actor 只能作为独立消融；
- 将证据网络最后一层初始化为接近零；
- 从较小但非零的 \(c_{\mathrm{op}}\) 开始；
- 主要通过 PPO 优势训练 \(B_{\phi_b}\)。

### 阶段三：联合微调

当 \(b\) 和 \(z\) 的分布稳定后，联合更新：

\[
\theta,\phi_b,\omega,
\]

并采用较小的证据网络学习率：

\[
\alpha_B\approx0.1\alpha_{\mathrm{actor}}.
\]

当前阶段仍不开放意见动力学参数学习。

---

## 12. 建议代码模块

```text
ConflictGraph
    纯函数式地识别当前冲突车辆并输出 edge mask

InteractionEncoder
    构造车辆对局部特征 chi_ij

OpinionEvidenceNet
    输出相对评分 ell_ij 和有界证据 b_ij

OpinionDynamics
    无状态地将 z_prev 与当前证据积分为 z_next

BaseActor
    根据局部物理观测输出基础动作分布

OpinionResidual
    将 z_ij 转换为动作均值残差

OpinionAugmentedActor
    组合 BaseActor 与 OpinionResidual

CentralizedCritic
    训练阶段估计全局价值

RecurrentRolloutBuffer
    保存连续序列、初始意见状态和 edge mask

MAPPOTrainer
    实现 PPO、截断时间反向传播和梯度隔离
```

其中：

- `OpinionDynamics` 的参数默认不可训练；
- `z_dense` 由 Stateful Collector 按 global agent ID 持有，不存在环境、
  `ConflictGraph` 或 `OpinionDynamics` 模块内；
- `OpinionEvidenceNet` 属于 Actor 参数组；
- 第一版 Critic 不读取意见状态；后续若读取则必须停止梯度；
- 执行阶段不调用 `CentralizedCritic`。

---

## 13. 后续修改必须保持的原则

1. \(b\) 是瞬时物理证据，\(z\) 是跨时间意见状态；
2. \(B_{\phi_b}\) 不读取 \(z\)；
3. \(b\) 必须经过冲突紧迫度、置信度和最大幅值门控；
4. 当前只学习 \(b\)，不学习意见动力学参数；
5. 意见必须通过显式、有限的动作残差影响策略；
6. 基础动作分支不能直接绕过或复制意见分支；
7. Critic 损失不能反向更新 \(B_{\phi_b}\)；
8. 训练必须保留连续序列并展开 \(z\)；
9. 执行阶段不使用全局状态或中央 Critic；
10. 不加入人工意见—动作标签或反事实教师；
11. 不将意见动力学替换为普通黑盒循环网络；
12. 所有冲突边必须具有明确的创建、掩码、衰减和重置逻辑。

---

## 14. 最终模型总结

完整模型可以概括为：

\[
\boxed{
\begin{aligned}
b_{ij}^t
&=
b_{\max}\rho_{ij}^tc_{ij}^t
\tanh\left(
\frac{
G_{\phi_b}(\xi_i^t,\xi_j^t)
-G_{\phi_b}(\xi_j^t,\xi_i^t)
}{T_b}
\right),\\[2mm]
z_{ij}^{t}
&=
z_{ij}^{t-1}
+\Delta t\,\eta_z
\left[
-\kappa_z z_{ij}^{t-1}
+\rho_{ij}^t\nu_z\tanh(\alpha_z z_{ij}^{t-1})
+b_{ij}^t
\right],\\[2mm]
q_{ij}^t
&=
\tanh(z_{ij}^t/z_0),\\[2mm]
\mu_i^t
&=
\mu_{i,\mathrm{base}}^t
+c_{\mathrm{op}}
\sum_j\bar\rho_{ij}^tq_{ij}^t\mathbf d_{ij}^t,\\[2mm]
(\theta,\phi_b)
&\leftarrow
\arg\max J_{\mathrm{MAPPO}}
-\lambda_{\mathrm{neutral}}\mathcal L_{\mathrm{neutral}}
-\lambda_{\mathrm{mag}}\mathcal L_{\mathrm{mag}}.
\end{aligned}
}
\]

系统的最终分工是：

\[
\boxed{
\begin{aligned}
\text{MARL 学习：}&\quad
\chi_{ij}^t\rightarrow b_{ij}^t,\\
\text{意见动力学完成：}&\quad
(b_{ij}^{0:t},z_{ij}^{t-1})\rightarrow z_{ij}^t,\\
\text{Actor 完成：}&\quad
(o_i^t,z_{ij}^t)\rightarrow a_i^t,\\
\text{任务回报评价：}&\quad
\text{这种意见形成过程是否改善联合驾驶行为}.
\end{aligned}
}
\]

核心不是让 \(b\) 直接替代策略，而是让 MARL 学习局部交互中的瞬时证据，再由意见动力学把这些证据转化为连续、稳定、具有记忆且可解释的协调意见。
