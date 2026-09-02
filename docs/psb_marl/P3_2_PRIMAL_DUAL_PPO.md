# PSB-MARL P3.2：Projected Primal-Dual Sequence PPO

P3.2 首次联合更新分岔 Actor 和两个安全对偶变量。它复用 P2 已验证的序列 PPO、
近端隐式梯度和 Base 锚定动作结构，不建立第二套策略优化器。

## 1. 拉格朗日回报

每轮 rollout 使用固定的对偶变量：

\[
r_t^{\mathrm L}=r_t-\lambda_b\|b_t\|^2/|\mathcal E|
-\lambda_{\mathrm{veh}}c_t^{\mathrm{veh}}
-\lambda_{\mathrm{lane}}c_t^{\mathrm{lane}}.
\]

`c_vehicle` 和 `c_lane` 与 P3.1 完全相同，分别为 CPA 冲突风险和车道裕度缺口，
并用真实碰撞指示兜底。对固定的两个 multiplier，三通道价值的该线性组合就是
拉格朗日价值，因此 Actor 仍可使用一个标量 GAE 和原序列 PPO。

## 2. 对偶更新

完成整轮 PPO 后，使用 detached rollout 均值执行一次投影上升：

\[
\lambda_k\leftarrow
\Pi_{[0,\lambda_{\max}]}
\left(\lambda_k+\eta_k(\bar c_k-d_k)\right).
\]

对偶变量不是神经网络参数，不进入 autograd，也不会通过 Critic 回归修改近端状态。
车道头在 P3.1 中只小幅优于常数基线，因此其学习率小于车辆对偶学习率。

## 3. 网络与部署

```text
local observation + conflict graph
        -> antisymmetric b -> proximal z -> longitudinal Actor residual
        -> sampled physical action

dense vehicle/lane costs -> projected dual controller
task reward + energy + fixed duals -> scalar Lagrangian GAE -> sequence PPO
```

P3.2 从 P3.1 中字节不变的 P2 Candidate Actor 和原 P2 增广标量 Critic warm-start。
P3.1 三通道 Critic作为已验证的目标定义与校准证据保存在运行目录中。训练后的 Actor
仍只保存为 `candidate_policy.pth`；`final_policy.pth` 保持 Base fallback，直至后续
多种子配对非劣门控通过。

## 4. 手动训练

```bash
python main_training.py \
  --config configs/psb_marl/p3_2_primal_dual_ppo.json
```

训练期间重点观察：

- `vehicle_cost_mean`、`lane_cost_mean` 及各自 constraint residual；
- `vehicle_multiplier`、`lane_multiplier` 不为 NaN 且保持在 `[0, 2]`；
- `sequence_state_replay_abs_error`、近端根残差和 sector violation；
- task episode reward 与真实碰撞率，不使用拉格朗日回报冒充环境性能。

训练完成后不要立即提升 Candidate。先将新的 run directory 发回，再执行配对长时域
测试和 Base fallback 门控。

## 5. P3.2-N：无量纲约束修正

首轮 P3.2 验证发现，车辆代价约为 `0.35`，车道代价约为 `0.008`。直接使用绝对
残差会使车道对偶更新比车辆通道小约两个数量级。P3.2-N 将每个约束除以其预算：

\[
g_k=\frac{\bar c_k}{d_k}-1,
\qquad
r_t^{\mathrm L}=r_t^{\mathrm{aug}}
-\sum_k\lambda_k\frac{c_{k,t}}{d_k}.
\]

正比例缩放约束不改变 CMDP 的可行域或 KKT 解，只改变 multiplier 的单位和数值条件。
预算使用 P3.1 训练种子上的 Base 统计锁定为车辆 `0.35`、车道 `0.0075`，不使用
P3.2 测试种子调参。对应配置为
`configs/psb_marl/p3_2_n_normalized_primal_dual_ppo.json`。

## 6. P3.2-C：动作子空间一致的约束

P2.1-U 明确将策略残差投影到纵向动作，转向均值与 Base 字节路径完全一致。因此车辆
冲突风险对当前策略可控，而几何车道裕度主要属于横向控制量。将不可控的车道裕度放入
对偶优化只会让 multiplier 发散并损害任务回报，不能产生可实现的下降方向。

P3.2-C 只对车辆冲突约束执行预算归一化 primal-dual PPO：

\[
\max_\theta J_R(\theta),\qquad
J_{\mathrm{veh}}(\theta)\le d_{\mathrm{veh}}.
\]

车道安全不被删除，而是由三个独立门控保持：Base 转向均值精确保留、车道碰撞统计
非劣、部署前 Base fallback。车道连续代价仍被记录为诊断量，但不产生错误的策略梯度。
若未来开放横向残差，才重新把它加入 `active_constraints`。对应配置为
`configs/psb_marl/p3_2_c_actuation_aligned_primal_dual_ppo.json`。

P3.2-C 的最终鲁棒性使用完全相同的超参数和 P3.1 parent，只改变训练随机种子；
seed 1、2 配置分别为 `p3_2_c_actuation_aligned_primal_dual_ppo_seed1.json`
和 `p3_2_c_actuation_aligned_primal_dual_ppo_seed2.json`。
