# A6-Action：学习首选动作，固定 AVOCADO 安全投影

## 1. 阶段目标

A6-Action 是在 A6-Y 之后建立的独立研究分支。A6-Y 只能通过修正启发式合作估计
\(y^H\) 间接影响意见状态、责任分配和最终动作，因此其策略空间仍受 AVOCADO 意见模型
约束。A6-Action 改为直接学习 Base Actor 动作分布的位置参数修正：

\[
(\mu_t^B,\sigma_t^B)=\pi_B(o_t),\qquad
\Delta\mu_t^A=f_\theta(x_t^{pair},\mu_t^B),
\]

\[
a_t^{preferred}\sim
\operatorname{TanhNormal}
(\mu_t^B+\Delta\mu_t^A,\sigma_t^B).
\]

其中 \(a_t^{preferred}\) 是速度和转角组成的首选自行车动作。AVOCADO 的启发式
\(y^H\)、注意力、非线性意见递推、OCA/道路约束和 TTC 屏障均保持固定，仅作为最终
可行性与安全层：

\[
a_t^{executed}
=\operatorname{Safety}_{AVOCADO}(s_t,a_t^{preferred}).
\]

因此 RL 不再通过 \(\Delta y\) 改写 AVOCADO 的意见，而是能够提出独立的首选动作；
AVOCADO 只在动作不满足冲突和道路约束时进行投影。

## 2. 交互动作网络

网络复用 A5 已验证的 14 维局部车辆对特征，不读取全局 agent 编号。每个有效车辆对先
经过共享编码器，再使用置信度加权均值和 masked maximum 聚合。该结构对邻车排列不变，
同时保留平均交互态势与最危险邻车信息。聚合结果与冻结 Base Actor 的
\(\mu_t^B\) 拼接，输出二维有界修正：

\[
\Delta\mu_t^A
=d_{max}\odot\tanh(f_\theta(c_t,\mu_t^B)).
\]

当前配置令速度和转角 loc 修正上界均为 2.0。这里的上界属于动作分布的未压缩 loc
空间，不直接等于 m/s 或 rad；最终物理动作仍由 `TanhNormal` 映射到环境动作上下界。
没有有效交互车辆对时，修正被显式置零，从而保留 Base Actor 的普通巡航行为。

## 3. 不可破坏的阶段合同

- Base Actor 完全冻结，训练前后参数哈希必须相同；
- AVOCADO 参数、启发式意见、OCA、道路约束和 TTC 屏障完全固定；
- Central Critic 可以更新，但只用于优势估计；
- 只训练 `InteractionActionNet`，PPO 不需要对 AVOCADO 或环境反向传播；
- 动作头最后一层零初始化，iteration 0 的策略分布严格等于 Base Actor；
- 未训练 A6-Action 与 A5 在同 seed 下逐物理步完全一致；
- 无有效车辆对时，动作修正必须严格为零；
- checkpoint 必须绑定 Base Actor 参数哈希和完整配置指纹。

零初始化合同为：

\[
\Delta\mu_t^A=0
\Longrightarrow
\pi_{A6\text{-}Action}=\pi_B
\Longrightarrow
a_t^{executed,A6\text{-}Action}=a_t^{executed,A5}.
\]

自动验收会逐步比较名义动作、投影前动作、执行动作、启发式估计、融合估计、意见、
注意力、邻接 mask 和 reset mask，所有项要求精确差值为零。

## 4. PPO 数据流

Rollout 保存当前观测、车辆对特征、置信度、pair mask、采样的首选动作、旧 log-prob、
reward、done、Critic value 和 loc 修正。更新时重新计算首选动作分布并使用普通 clipped
PPO：

\[
\mathcal L_{actor}
=-\mathcal L_{PPO}
-\lambda_H\mathcal H
+\lambda_A\|\Delta\mu_t^A\|_2^2.
\]

正则项只在存在有效交互的 agent 上计算。动作修正直接进入策略分布，所以梯度路径为
`log_prob -> loc correction -> InteractionActionNet`，不依赖 AVOCADO 的可微性。

## 5. 代码与配置

- 正式配置：`configs/avocado_marl/a6_action.json`
- 短流程配置：`configs/avocado_marl/a6_action_pilot.json`
- 策略与执行桥：`utilities/avocado_marl/a6_action_policy.py`
- PPO、checkpoint 与评估：`utilities/avocado_marl/a6_action_trainer.py`
- 训练入口：`main_training_avocado_marl_a6_action.py`
- 测试入口：`main_testing_avocado_marl_a6_action.py`
- 自动测试：`tests/avocado/test_a6_action.py`

先验证零初始化闭环合同：

```bash
conda run -n sigmarl-nod python main_testing_avocado_marl_a6_action.py \
  --verify-zero --max-steps 4
```

运行一轮 pilot：

```bash
conda run -n sigmarl-nod python main_training_avocado_marl_a6_action.py \
  --config configs/avocado_marl/a6_action_pilot.json
```

训练正式配置：

```bash
conda run -n sigmarl-nod python main_training_avocado_marl_a6_action.py
```

评估最新正式 checkpoint：

```bash
conda run -n sigmarl-nod python main_testing_avocado_marl_a6_action.py \
  --max-steps 200 --scenario CPM_mixed --seed 0
```

加入 `--render` 可实时查看首选动作、执行动作和 loc 修正。

## 6. 当前验收结论与边界

截至 2026-08-30，策略级严格回退、冻结 Actor 梯度隔离、四步 A5 闭环精确等价、pilot
PPO 参数更新和 checkpoint 重载评估均已通过。pilot 只证明训练链有效，不构成性能结论。

下一实验必须在相同训练预算和训练 seed 下比较 A5、A6-Y 与 A6-Action，并至少报告
碰撞、路线完成、奖励、实测速度、路径误差、AVOCADO 动作干预率、TTC 屏障率和
\(\Delta\mu^A\) 的幅值/饱和率。如果 A6-Action 产生较大的首选动作变化但长期被
AVOCADO 投影消除，应将其视为策略与安全层冲突的证据，而不是简单增加网络容量或训练
轮数。
