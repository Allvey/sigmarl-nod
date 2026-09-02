# PSB-MARL P3.1：Base-relative Differential Critic

P3.1 只训练中央 Critic，P2.1-U Actor、近端分岔层和对偶变量全部冻结。其目的不是
再次改变车辆动作，而是用 P3.0 已验证的 Candidate/Base 配对数据建立低方差、可分解的
相对价值基线。第一次 P3.1 试跑证明了标量 reward 头可学习，但直接使用 0/1 碰撞
事件会使安全头退化：无车道碰撞的数据中，车道目标方差严格为零。因此当前版本使用
“连续安全残差 + 真实碰撞兜底”，碰撞率仍只作为最终性能指标。

## 1. 三通道相对价值

P3.1 不把安全代价用人工权重混入一个标量奖励，而是同时拟合：

\[
\boxed{
\Delta V_\psi=
\left(
\Delta V_{\widetilde r},
\Delta V_{\mathrm{veh}},
\Delta V_{\mathrm{lane}}
\right).
}
\]

其 Monte Carlo 监督目标为：

\[
\Delta G_t^k=G_{t,\mathrm{Candidate}}^k-G_{t,\mathrm{Base}}^k,
\quad
k\in\{\widetilde r,\mathrm{veh},\mathrm{lane}\}.
\]

其中任务通道为：

\[
\widetilde r_t=r_t-\lambda_b
\frac{1}{|\mathcal E|}\sum_{i<j}b_{ij,t}^2.
\]

车辆连续安全代价复用无通信冲突图：

\[
\omega_{ij,t}=m_{ij,t}\,u_{ij,t}\,q_{ij,t},\qquad
c_{i,t}^{\mathrm{veh}}
=\max\left\{\mathbf 1_{\mathrm{collision}},\max_j\omega_{ij,t}\right\}.
\]

这里 `u` 是 CPA urgency，`q` 是几何 confidence，`m` 是本地可观测冲突边掩码。
因此车辆尚未碰撞、但预计接近时间和最近距离已经恶化时，Critic 就能得到连续监督。

车道连续安全代价为标准的裕度缺口：

\[
c_{i,t}^{\mathrm{lane}}
=\max\left\{
\mathbf 1_{\mathrm{collision}},
\left[\frac{d_{\mathrm{safe}}-d_{i,t}}{d_{\mathrm{safe}}}\right]_{[0,1]}
\right\}.
\]

`d_i` 是车辆外形到左右边界的最小距离，环境已按 `3 * lane_width` 归一化；
`CPM_mixed` 的 `d_safe=0.07` 与环境原有 near-boundary 阈值一致。

Candidate 和 Base 分别按照自己的 `done/terminated` 边界计算回报，再求差。不能先对
即时差分做一次共享终止掩码的折扣累积，因为两个闭环可能在不同时间结束 episode。

P3.2 将使用：

\[
A_t^{\mathrm L}
=A_t^{\Delta r}
-\lambda_{\mathrm{veh}}A_t^{\Delta c_{\mathrm{veh}}}
-\lambda_{\mathrm{lane}}A_t^{\Delta c_{\mathrm{lane}}},
\]

从而把任务优化与两个安全约束保持为清晰的 primal-dual 结构。

## 2. 网络结构

```text
Candidate observation ─┐
                       ├─ shared NodeEncoder ── h_i ─────────────┐
Base observation ──────┘                                         │
                                                                 │
(h_i, h_j, z_ij, |z_ij|, edge mask)                              │
              └─ shared EdgeEncoder ─ mean_j(message_ij) ────────┤
                                                                 │
mean_i(node_i + message_i) ── global context ────────────────────┤
                                                                 ▼
                                                    shared 3-head critic
                                        (Delta reward, vehicle risk, lane margin)
```

节点、边和输出头在所有智能体之间共享，边消息只沿有效冲突边聚合。均值聚合保证网络：

- 对智能体重新编号保持置换等变；
- 不依赖训练时固定的智能体数量；
- 显式读取完整的 `z_ij` 和边生命周期掩码；
- 只在 CTDE 训练中使用，执行阶段完全删除。

三个物理目标的量级差异很大，因此网络保存训练集目标的 center/scale，损失在归一化
残差上计算 Huber loss，输出仍恢复为原始物理单位。

## 3. 冻结与部署保证

配置强制：

```text
actor_learning_enabled = false
dual_learning_enabled  = false
```

P3.1 运行目录中：

- `candidate_policy.pth` 与通过 P3.0/P2.2-R 的 Actor 字节完全相同；
- `candidate_critic.pth` 是新的三通道差分 Critic；
- `paired_critic_dataset.pt` 保存去梯度后的紧凑训练/validation 数据；
- `source_p2_critic.pth` 保留 P2 Critic，仅用于追溯；
- `final_policy.pth` 和 `final_critic.pth` 仍是原始 Base fallback。

所以 P3.1 不可能改变 rollout 动作，当前“不弱于 Base”的保障不会因 Critic 校准而失效。

## 4. 锁定的数据划分

配置文件为：

```text
configs/psb_marl/p3_1_differential_critic.json
```

默认协议：

```text
scenario:          CPM_mixed
training seeds:    701 702 703 704
validation seeds:  705 706
episodes/seed:     4
max steps:         600
critic epochs:     50, early stopping patience 8
```

训练种子与 validation 种子必须互斥。训练 certification 同时要求：

- unseen validation 总 Huber loss 至少比“始终预测训练目标均值”的常数 Critic改善 1%；
- 三个通道的 `target_std >= 1e-4`，禁止零方差安全头被总损失掩盖；
- 三个通道各自的 normalized Huber 都不高于常数 Critic，保证每个头在实际训练目标上
  至少不弱于均值预测器。

`explained_variance` 继续作为诊断量报告，但不单独决定门控。它会先去除误差均值，可能
出现“Huber、MAE 和 RMSE 均改善，但 explained variance 轻微为负”的情况；用它替代
实际优化目标会错误拒绝一个不劣的 Critic 头。

## 5. 手动训练

```bash
python main_training.py \
  --config configs/psb_marl/p3_1_differential_critic.json
```

该命令会进行 12 次固定策略 rollout（6 个种子 × Candidate/Base），然后只优化 Critic。
它不会执行 PPO Actor 更新。

训练完成后先检查：

```text
<P3_1_RUN_DIR>/p3_1_certification.json
```

必须满足：

```json
{
  "passed": true,
  "actor_bytes_unchanged": true,
  "actor_learning_enabled": false,
  "dual_learning_enabled": false
}
```

并确认：

```text
achieved_relative_improvement >= 0.01
channel_quality.passed = true
```

## 6. 手动 holdout 验证

使用未参与训练和 early stopping 的新种子：

```bash
python main_testing.py \
  --config configs/psb_marl/p3_1_differential_critic.json \
  --run-dir <P3_1_RUN_DIR> \
  --checkpoint <P3_1_RUN_DIR>/candidate_policy.pth \
  --scenario CPM_mixed \
  --max-steps 600 \
  --episodes 4 \
  --seeds 711 712 \
  --no-render \
  --compare-base
```

`p3_1_manual_validation.json` 顶层必须满足：

```text
passed = true
critic_passed = true
deployment = base_fallback
```

并同时检查：

- `critic_relative_improvement >= 0.01`；
- 所有 `structural_checks` 为 `true`；
- `critic_channel_quality.passed = true`；
- 每个通道的 `loss_noninferiority_passed = true`；
- 三个通道的 MAE、RMSE、explained variance 和非零样本符号准确率；
- `paired_comparisons` 只作为未训练 Actor 的复核，不在两种子上重新宣称统计非劣。

只有训练 certification 与新种子 manual validation 都通过，才进入 P3.2 Actor + dual
更新。若 Critic 未超过常数基线，应先修改特征、容量或采样覆盖，不能直接打开 Actor。
