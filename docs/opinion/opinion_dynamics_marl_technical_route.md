# AVOCADO 与 MARL：学习邻车合作估计的有界修正

> 文档角色：后续 AVOCADO-MARL 方法的最高级理论真源
> 当前基线：A0-A2 严格全向 AVOCADO；A3.4 AVOCADO-KB 道路确定性基线；
> A4 Base-MAPPO 与固定 AVOCADO 动作级耦合；A5 零修正网络等价接口；
> A6 冻结 Base Actor 的单步截断 PPO 修正学习
> 目标方法：保留 AVOCADO 的注意力、合作估计和非线性意见动力学，仅由 MARL 学习
> 启发式合作估计 \(y^H\) 的有界修正 \(\Delta y^{RL}\)
> 对齐日期：2026-08-29

## 0. 当前实现与后续目标的边界

A0-A2、A3.4、A4和A5已完成；A6单步训练链已完成代码和pilot验证，正式多seed性能
实验尚未执行。前三者继续作为不可混淆的无学习基线，A4作为动作级耦合基线，A5作为
零修正接口基线：

- A0-A2 使用圆盘单积分器和二维速度动作，验证 AVOCADO 官方几何与意见递推；
- A3.4 将其接入 SigmaRL `road_traffic`、RK4 自行车动力学、道路速度锥、互补责任、
  速度连续性目标和可审计 TTC 屏障；
- A3.4 不使用 MAPPO、EvidenceNet 或任何学习参数。
- A4由冻结的Base-MAPPO输出名义自行车动作，固定A3安全链输出执行动作；仍然没有
  `YCorrectionNet`，且 \(\Delta y^{RL}=0\)。
- A5加入冻结的严格零输出 `YCorrectionNet` 和融合接口；完整闭环严格退化为A4，
  AVOCADO内部的估计与非线性递推严格退化为A3。

现有 M2-M9 Opinion-MARL 代码实现了另一条历史原型：网络输出加性证据 `b`，再进入
重新设计的 OpinionDynamics。该代码和对应文档保留用于复现及消融，但它不再是后续
AVOCADO-MARL 的目标理论路线。新路线不得把原 AVOCADO 的固定偏置 \(b\) 重新命名为
学习证据。

新路线的零修正条件必须使完整闭环精确退化为A4，并使AVOCADO子模块精确退化为A3：

\[
\Delta y^{RL}=0
\quad\Longrightarrow\quad
y^{F}=y^{H}
\quad\Longrightarrow\quad
\text{原 AVOCADO 意见递推不变},\qquad
\text{A5执行动作}=\text{A4执行动作}.
\]

## 1. 方法目标与模块分工

完整方法将三个层次明确分开：

```text
局部物理状态
   ├── Base-MAPPO Actor ─────────────────────────→ 名义驾驶动作
   │
   └── AVOCADO 实时估计
          TTC → 注意力 A
          邻车速度变化 + 上一拍 VO 修正 → y^H
                                      │
                                      ├── YCorrectionNet → Δy^RL
                                      │
                                      ↓
                             y^F = bounded(y^H + Δy^RL)
                                      ↓
                         原 AVOCADO 非线性动力学 → z
                                      ↓
                  有界纵向策略残差 + OCA 避让责任
                                      ↓
                  道路/OCA 联合求解 + 自行车适配 + TTC 屏障
                                      ↓
                                   env.step()
```

各模块职责如下：

| 模块 | 职责 | 是否学习 |
|---|---|---:|
| AVOCADO 注意力 \(A\) | 根据 TTC 表示冲突紧迫程度 | 否 |
| 启发式估计 \(y^H\) | 从实际速度变化估计邻车合作行为 | 否 |
| `YCorrectionNet` | 修正自行车运动学、道路控制等造成的估计偏差 | 是 |
| AVOCADO 意见状态 \(z\) | 积累、遗忘并非线性强化合作判断 | 否 |
| Base Actor | 学习正常路径驾驶和任务级动作 | 是 |
| Opinion Residual | 让 \(z\) 显式、有限地影响策略分布 | 否 |
| OCA/道路求解器 | 将名义意图投影到几何可行速度 | 否 |
| Central Critic | 集中训练阶段估计价值 | 是 |

## 2. AVOCADO 固定实时量

### 2.1 注意力

对有向车辆对 \((i,j)\)，使用当前圆盘安全半径、相对位置和实测速度计算首次碰撞时间
\(\tau_{ij}^t\)。注意力严格采用 AVOCADO 官方实现的离散滤波：

\[
A_{ij}^{t}
=
(1-\delta)A_{ij}^{t-1}
+\delta\tanh\left(\frac{\kappa}{\tau_{ij}^{t}}\right).
\]

\(A\) 不由 MARL 输出，也不设可训练参数。没有未来碰撞时刺激为零；碰撞迫近时注意力
趋近 1。

### 2.2 启发式邻车合作估计

当前邻车实测速度变化为：

\[
\Delta\mathbf v_j^t
=
\mathbf v_j^t-\mathbf v_j^{t-1}.
\]

令 \(\mathbf u_{ij}^{t-1}\) 为上一拍针对车辆 \(j\) 的 VO 最小修正，则：

\[
r_{ij}^{t}
=
\begin{cases}
\dfrac{
|\Delta\mathbf v_j^{t\mathsf T}\mathbf u_{ij}^{t-1}|
}{
\|\mathbf u_{ij}^{t-1}\|^2
},
& \|\mathbf u_{ij}^{t-1}\|^2>\epsilon_u,\\
0,&\text{otherwise},
\end{cases}
\]

\[
y_{ij}^{H,t}
=
\tanh\left[
\epsilon_y\left(r_{ij}^{t}-\frac12\right)
\right].
\]

该估计仍按 AVOCADO 的因果顺序使用“当前速度变化 + 上一拍 VO 修正”，不使用未来
动作或轨迹。\(y^H\) 在自行车环境中会混入转向、非完整运动学、动作适配和安全屏障的
影响，这正是学习修正存在的原因。

## 3. MARL 学习的有界修正

### 3.1 输入特征

`YCorrectionNet` 只读取本地可获得的物理量和 AVOCADO 当前派生量：

\[
\chi_{ij}^t=
[\Delta p,\Delta v,v_i,v_j,
\sin\Delta\psi,\cos\Delta\psi,
\tau,d_{CPA},A,y^H,\|u^{t-1}\|,m].
\]

第一版仍只处理原始局部观测中的两个最近邻。禁止输入：

- 意见状态 \(z\) 或由其导出的合作责任；
- 未来真实轨迹、未来动作或碰撞结果；
- 全局车辆 ID、中央 Critic 隐状态；
- 其他车辆不可观测的策略参数。

禁止读取 \(z\) 是为了避免 \(z\rightarrow\Delta y\rightarrow z\) 的内部正反馈。

### 3.2 修正公式

共享网络输出方向性的原始修正：

\[
\widetilde{\Delta y}_{ij}^{t}
=
\tanh\left(
H_{\phi_y}(\chi_{ij}^t)/T_y
\right).
\]

最终修正为：

\[
\Delta y_{ij}^{RL,t}
=
\lambda_y
m_{ij}^t
c_{ij}^{sense,t}
\widetilde{\Delta y}_{ij}^{t},
\]

\[
\boxed{
y_{ij}^{F,t}
=
\operatorname{clip}
\left(
y_{ij}^{H,t}+\Delta y_{ij}^{RL,t},
-1,1
\right)
}
\]

其中：

- \(m_{ij}\) 是有效车辆对 mask；
- \(c^{sense}\in[0,1]\) 是由感知距离等物理量计算的置信度；
- \(\lambda_y\) 是固定的最大修正幅值，第一版建议从 `0.1` 开始，硬上限不超过 `0.5`；
- 网络最后一层近零初始化，使初始策略与 A3 近似相同；严格零初始化模式用于数值等价测试。

这里不要求 \(\Delta y_{ij}=-\Delta y_{ji}\)。原始 \(y_{ij}\) 表示“车辆 \(i\) 观察到
车辆 \(j\) 的合作程度”，两车可能同时合作或同时不合作，强制反对称会破坏该语义。

## 4. 保持不变的 AVOCADO 非线性动力学

融合估计进入原 AVOCADO 方程：

\[
\boxed{
z_{ij}^{t+1}
=
z_{ij}^{t}
+\Delta t
\left[
-d z_{ij}^{t}
+d A_{ij}^{t}
\tanh\left(
a z_{ij}^{t}+c y_{ij}^{F,t}
\right)
+b_0
\right]
}
\]

第一版固定 A3 参数：

\[
d=2,\quad a=0.3,\quad c=0.7,\quad
\kappa=14.15,\quad\epsilon_y=3.22,\quad
\delta=0.57,\quad b_0=0.
\]

MARL 不学习 \(d,a,c,\kappa,\epsilon_y,\delta,b_0\)。这样可以保留 AVOCADO 的
解释性、零修正退化性质和稳定性分析，避免策略、估计器与动力学同时漂移。

当车辆对失效但车辆未重置时，自强化与学习修正关闭，旧意见按既定生命周期处理；
车辆、环境或 global-ID 映射重置时，必须同时清除入边和出边的 \(A,y,u,z\) 状态。

## 5. 意见如何影响 MARL 与 OCA

### 5.1 可微的策略残差

将意见有界化：

\[
q_{ij}^t=\tanh(z_{ij}^t/z_0).
\]

按物理冲突权重归一化聚合，只修改 Actor 的纵向动作均值：

\[
\Delta\mu_{i,op}^t
=
c_{op}
\sum_j\bar w_{ij}^t q_{ij}^t,
\qquad
\bar w_{ij}^t
=
\frac{A_{ij}^tm_{ij}^t}
{\epsilon+\sum_k A_{ik}^tm_{ik}^t}.
\]

\[
\mu_i^t
=
\mu_{i,base}^t
+[\Delta\mu_{i,op}^t,0].
\]

第一版不让意见直接修改转向分量。该残差提供
\(\phi_y\rightarrow\log\pi\) 的可微路径，否则仅把 \(z\) 用在非可微 OCA 投影中将
无法有效训练 `YCorrectionNet`。

### 5.2 OCA 责任

意见仍按 AVOCADO 映射为对邻车合作程度的估计：

\[
s_{ij}=\operatorname{clip}\left((z_{ij}+1)/2,0,1\right).
\]

道路版本使用互补责任：

\[
R_{ij}
=
\frac{1-s_{ij}}
{(1-s_{ij})+(1-s_{ji})},
\qquad
R_{ji}=1-R_{ij}.
\]

当分母数值上接近零时，两车责任都取 \(0.5\)。因此互补和严格为 1，而不是通过在
分母中加入 \(\epsilon\) 得到近似互补。

MARL 名义动作先转换为世界坐标首选速度，再进入 OCA 半平面、道路速度锥、最大速度圆
和 A3.4 连续性目标的联合求解。求解后再做自行车动作适配和 TTC 屏障。屏障干预必须
单独记录，不能计为学习意见的贡献。

## 6. 强化学习与梯度边界

训练阶段采用 CTDE。Base Actor 和 `YCorrectionNet` 属于策略侧，Central Critic 只在
训练时估计价值：

\[
\boxed{
\phi_y
\rightarrow
\Delta y^{RL}
\rightarrow
y^F
\rightarrow
z
\rightarrow
q
\rightarrow
\Delta\mu_{op}
\rightarrow
\log\pi
\rightarrow
\widehat A_t
}
\]

策略损失可写为：

\[
\mathcal L_{actor}
=
-\mathcal L_{PPO}
-\lambda_H\mathcal H
+\lambda_{corr}\mathbb E[(\Delta y^{RL})^2]
+\lambda_{sat}\mathbb E[\max(|y^F|-y_{soft},0)^2].
\]

修正正则只用于鼓励“能由启发式解释时不修正”，不使用人工合作标签。价值损失不得
反向更新 Base Actor 或 `YCorrectionNet`。执行阶段不调用 Central Critic，也不会在线
更新网络参数。

## 7. A6采用单步截断 PPO

完整时间梯度原本可以写成：

\[
\Delta y^t\rightarrow z^t\rightarrow z^{t+1}\rightarrow\cdots,
\]

但前期序列 PPO 实验稳定性和效果较差，因此 A6 主路线暂不跨物理步反向传播。每个
样本保存更新前的 (z_t)，并把它视为停止梯度的控制器状态；当前步重新计算：

\[
\Delta y_t^{RL}\rightarrow y_t^F\rightarrow z_{t+1}
\rightarrow\Delta\mu_{op,t}\rightarrow\log\pi_t.
\]

动作均值必须使用当前新计算的 (z_{t+1})。如果使用更新前的 (z_t)，当前
\(\Delta y_t^{RL}\) 不影响当前动作，单步 PPO 将无法为 `YCorrectionNet` 提供策略梯度。

普通 Rollout Buffer 保存：

- 当前步的14维局部车辆对特征；
- 更新前的 \(z_t\)、注意力 \(A_t\)、有效边 mask 和 reset mask；
- 启发式 \(y^H\)、融合 \(y^F\) 与 \(\Delta y^{RL}\)；
- 名义动作、执行动作、旧策略 log-prob；
- episode 和单车重置边界。

样本可按普通 PPO minibatch 随机打乱。该方案忽略修正对更远期意见状态的时间梯度，
但保留当前步的可微因果链，优先验证修正学习本身。序列 PPO 只保留为后续可选消融，
不再是 A6 的前置条件。环境、自行车动力学与 OCA 求解器仍无需可微。

## 8. 分阶段、每步可验收的实施路线

### A4：MARL 名义动作接入固定 A3 安全链

实现状态：已于2026-08-28完成。入口为 `main_testing_avocado_marl.py`，配置为
`configs/avocado_marl/a4_base_avocado.json`，动作桥接和审计指标位于
`utilities/avocado_marl/`。

目标：不增加学习修正，先证明 Base-MAPPO 名义动作能够通过 A3 的世界速度转换、OCA、
道路约束、自行车适配和 TTC 屏障闭环运行。

固定：

\[
\Delta y^{RL}=0,\qquad y^F=y^H.
\]

验收：

- 实时可视化中 MARL 名义动作和执行动作均可观察；
- 改变 MARL 名义动作会可预测地改变无冲突轨迹，证明安全层没有完全接管策略；
- 与 Base-MAPPO、A3.4、Base-MAPPO+固定 ORCA 同种子比较；
- 动作有限、无状态串扰、reset 后 A3 状态清零。

### A5：零修正网络等价性

实现状态：已于2026-08-29完成。配置、实时入口和逐步等价验证分别位于
`configs/avocado_marl/a5_zero_correction.json`、`main_testing_avocado_marl_a5.py` 和
`utilities/avocado_marl/`。

加入 `YCorrectionNet`、融合器和序列状态接口，但冻结网络并令输出严格为零。

验收：

- A5 与 A4 在同 seed 下逐步满足
  \(y^F=y^H\)、\(z_{A5}=z_{A4}\)、执行动作一致；
- 网络参数冻结，测试与诊断入口能完整运行；
- 保存完整的 \(A,y^H,\Delta y,y^F,z\) 诊断序列。

### A6：只训练有界 \(\Delta y\)

实现状态：已完成单步截断PPO、显式Base checkpoint绑定、训练/评估入口、断点恢复和
审计指标；当前只通过短pilot，不能据此宣称性能优于A5。

冻结 Base Actor、AVOCADO 参数、残差方向与安全层，仅训练 `YCorrectionNet`；Critic 可
独立更新以提供优势估计。采用第7节的单步截断 PPO，不进行跨步 BPTT。

验收：

- 修正始终满足配置上界，非有效车辆对严格为零；
- 修正网络不读取 \(z\)，梯度只能来自 PPO Actor loss；
- 同预算、多 seed 比较 A5 与 A6，报告安全、效率和平滑性；
- 不能只报告最终奖励，必须报告修正幅值、饱和率、符号切换率以及冲突窗口指标。

### A7：Actor、\(\Delta y\) 与 Critic 联合训练

从零训练是正式主路线：Base Actor、`YCorrectionNet`、Critic 随机初始化，在与原始
Base-MAPPO 相同的总采样预算内联合优化。A6 warm start 和已训练 Base 初始化只作为
稳定性消融，不累加到正式主结果的预算中。

验收：

- Actor、修正网络、Critic 使用独立 optimizer 参数组和梯度裁剪；
- AVOCADO 动力学参数保持无梯度；
- checkpoint 完整恢复模型、optimizer、调度器、RNG 和训练计数；
- 多 seed 置信区间优于或不劣于必要基线。

### A8：最终消融与归因

至少比较：

1. Base-MAPPO；
2. 固定 ORCA-KB；
3. A3.4 启发式 AVOCADO-KB；
4. Base-MAPPO + 固定 OCA/道路安全层；
5. A5：MARL + 启发式 \(y^H\)，零学习修正；
6. A6/A7：MARL + \(y^H+\Delta y^{RL}\)；
7. 去掉 \(y^H\)、仅使用网络估计的消融；
8. 历史 EvidenceNet-\(b\) 原型，仅作为方法差异消融。

只有第 6 项在相同观测、训练预算、随机种子和安全层设置下稳定改善，才能把收益归因于
“学习启发式 \(y\) 的有界修正”。

## 9. 验证指标

除奖励、车辆碰撞、车道事件、路线完成率、实测速度、路径误差和 TTC 屏障率外，新增：

- \(y^H\)、\(\Delta y^{RL}\)、\(y^F\)、\(z\) 的均值、P95、最大值和直方图；
- 修正幅值饱和率、修正符号切换率；
- 启发式与融合估计的差异 \(|y^F-y^H|\)；
- 冲突窗口内的转角变化 P95、转向反向率和停止动作率；
- 名义动作与 OCA/屏障后执行动作的差异；
- OCA 不可行率和互补责任误差 \(|R_{ij}+R_{ji}-1|\)；
- 不同 TTC 分箱下的修正、意见和控制行为。

全时域 P95 可能掩盖稀疏冲突阶段的抖动，因此冲突窗口指标是硬性报告项。

## 10. 建议配置结构

新配置不复用旧 `opinion.evidence` 字段，避免把两种方法误加载：

```json
{
  "avocado_marl": {
    "y_correction": {
      "enabled": true,
      "hidden_sizes": [128, 128],
      "temperature": 1.0,
      "maximum_absolute_correction": 0.1,
      "hard_limit": 0.5,
      "zero_initialize_output": true,
      "magnitude_regularization": 0.001,
      "saturation_regularization": 0.001
    },
    "sequence_ppo": {
      "enabled": true,
      "chunk_length": 32,
      "train_y_correction": true
    }
  }
}
```

加载器必须拒绝同时启用旧 `opinion.evidence` 和新 `avocado_marl.y_correction`，防止在
同一实验中混合两套不同语义的动力学。

## 11. 必须保持的原则

1. \(A\) 仍由 TTC 实时计算，MARL 不得学习或覆盖注意力；
2. \(y^H\) 始终保留，网络只输出有界修正 \(\Delta y^{RL}\)；
3. \(b_0\) 保持 AVOCADO 固定偏置语义，第一版设为 0；
4. 零修正必须逐步退化为启发式 AVOCADO；
5. 修正网络不得读取 \(z\)、未来真值或中央 Critic 状态；
6. 不强制 \(\Delta y_{ij}\) 与 \(\Delta y_{ji}\) 反对称；
7. AVOCADO 动力学参数固定且不可训练；
8. 意见通过显式有界纵向残差影响策略，保证 PPO 梯度可达修正网络；
9. OCA、道路速度锥、速度界和 TTC 屏障继续作为可审计安全链；
10. 训练必须保留连续序列、车辆身份映射和正确 reset；
11. Critic loss 不得更新 Actor 或修正网络；
12. 执行阶段只使用局部物理量和本地历史，不使用中央 Critic；
13. 历史 EvidenceNet-\(b\) 结果不得冒充新 \(y\)-correction 方法结果；
14. 所有性能结论必须来自相同预算、多 seed 和明确消融。

## 12. 最终模型总结

\[
\boxed{
\begin{aligned}
A_{ij}^{t}
&=(1-\delta)A_{ij}^{t-1}
+\delta\tanh(\kappa/\tau_{ij}^{t}),\\
y_{ij}^{H,t}
&=\tanh\left[\epsilon_y(r_{ij}^{t}-1/2)\right],\\
\Delta y_{ij}^{RL,t}
&=\lambda_y m_{ij}^tc_{ij}^{sense,t}
\tanh(H_{\phi_y}(\chi_{ij}^t)/T_y),\\
y_{ij}^{F,t}
&=\operatorname{clip}(y_{ij}^{H,t}+\Delta y_{ij}^{RL,t},-1,1),\\
z_{ij}^{t+1}
&=z_{ij}^{t}+\Delta t
\left[-dz_{ij}^{t}
+dA_{ij}^{t}\tanh(a z_{ij}^{t}+c y_{ij}^{F,t})+b_0\right],\\
\mu_i^t
&=\mu_{i,base}^t+[\Delta\mu_{i,op}^t,0],\\
(\theta,\phi_y,\omega)
&\leftarrow\operatorname{MAPPO}
\quad\text{while AVOCADO dynamics remain fixed.}
\end{aligned}
}
\]

该方案的核心不是让 MARL 替代 AVOCADO 的在线估计，而是让 MARL 在长期任务回报下，
对自行车道路环境中不再完全可靠的启发式合作估计做有限、可回退、可审计的修正。
