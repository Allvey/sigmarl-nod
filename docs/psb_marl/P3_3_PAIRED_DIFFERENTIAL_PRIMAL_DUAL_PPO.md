# PSB-MARL P3.3：Paired Differential Primal-Dual PPO

P3.3 修复 P3.2 的核心信噪比问题：Actor 不再从绝对 Candidate 回报学习，而是从
同随机数 Candidate/Base 回报差学习。近端分岔层、纵向动作投影、车辆安全对偶变量和
Base fallback 均保持不变。

## 1. 配对控制变量

每轮使用两个独立环境，分别执行当前 Candidate 和冻结 Base。两个 collector 在每轮
开始时使用相同的确定性 seed，并强制 reset；随后以锁步方式逐物理步推进，每一步也
使用相同的派生随机 seed，因此共享初始交通状态和对应的随机扰动：

\[
\Delta G_t^R=G_{t,\mathrm{cand}}^R-G_{t,\mathrm{base}},
\qquad
\Delta G_t^c=G_{t,\mathrm{cand}}^c-G_{t,\mathrm{base}}^c.
\]

Base 回报与 Candidate 参数无关，减去它不改变期望策略梯度，只作为控制变量降低场景
随机性。两条物理轨迹允许因动作不同而分离，但 episode 边界必须一致：任一侧结束时，
两侧同时形成联合 `done`；真正结束的一侧保留原 `terminated/truncated`，仍在运行的一侧
只标记为人工 `truncated`，不会伪造物理终止。随后仅重置受影响的并行环境，并在两侧
使用相同的新 seed。这样每个固定时间位置的差分回报都来自同一个配对 episode。

## 2. 差分拉格朗日 Advantage

P3.1 三通道 Critic 从离线校准 checkpoint warm-start，并在 P3.3 中在线更新。P3.2-C
只激活预算归一化的车辆约束，因此：

\[
\Delta V_t^L
=\Delta V_t^R
-\frac{\lambda_{\mathrm{veh}}}{d_{\mathrm{veh}}}
 \Delta V_t^{\mathrm{veh}},
\]

\[
A_t^\Delta=\Delta G_t^L-\Delta V_t^L.
\]

Advantage 在 environment/time 维上按 agent 分别标准化，再输入原 P2 sequence PPO。
Critic 使用 P3.1 固定的通道尺度执行 normalized Huber 回归。Critic 输入、目标和近端
状态全部 detached，因此 Critic 回归不会反向修改 Actor 或分岔状态；Actor 梯度仍只经
PPO log-prob、纵向残差、branch adapter 和隐式近端层传播。

## 3. 网络结构

```text
Candidate observation + conflict graph
        -> antisymmetric b -> proximal z -> longitudinal residual -> action
        -> Candidate rollout --------------------┐
                                                  ├-> delta returns
same CRN seed -> frozen Base -> Base rollout -----┘

either side done -> union boundary -> truncate counterpart
                 -> common-seed paired reset -> next paired episode

(Candidate obs, Base obs, z, edge mask)
        -> online P3.1 differential critic -> delta V_R, delta V_vehicle

delta return - delta value -> normalized A_delta -> sequence PPO -> Actor
Candidate vehicle cost     -> projected dual update -> lambda_vehicle
```

## 4. 产物与部署

- `candidate_policy.pth`：P3.3 Actor；
- `candidate_differential_critic.pth`：在线更新后的三通道 Critic；
- `source_differential_critic.pth`：字节不变的 P3.1 来源；
- `candidate_critic.pth`：冻结标量 Critic，仅用于旧产物兼容；
- `p3_dual_state.pt`：车辆 multiplier 与预算；
- `final_policy.pth`、`final_critic.pth`：仍为字节不变的 Base fallback。

训练完成不会自动部署 Candidate。

## 5. 手动训练

```bash
python main_training.py \
  --config configs/psb_marl/p3_3_paired_differential_primal_dual_ppo.json
```

训练期间优先观察以下指标，而不是绝对 `episode_reward_mean`：

- `paired_delta_reward_step_mean`；
- `paired_delta_vehicle_risk_step_mean`；
- `paired_episode_boundaries_exact` 必须为 `true`；
- `paired_post_reset_observation_max_abs_error` 必须不超过 `1e-6`；
- `paired_candidate/base_synthetic_truncation_count` 用于诊断哪一侧先结束；
- `paired_raw_advantage_mean/std`；
- `paired_training_advantage_mean/std`，标准化后应接近 `0/1`；
- `online_differential_critic_mae`；
- `vehicle_cost_mean`、`vehicle_multiplier`；
- `sequence_approx_kl`、`sequence_state_replay_abs_error`；
- `rollout_sector_bound_max_violation` 和近端根残差。

两个 collector 会增加 rollout 成本和内存，但不增加 PPO epoch 数；当前项目主要耗时仍
来自 60 个优化 epoch。

## 6. 手动验证

训练结束后把 `<P3_3_RUN_DIR>` 替换成实际目录：

```bash
python main_testing.py \
  --config configs/psb_marl/p3_3_paired_differential_primal_dual_ppo.json \
  --run-dir <P3_3_RUN_DIR> \
  --checkpoint <P3_3_RUN_DIR>/candidate_policy.pth \
  --scenario CPM_mixed \
  --max-steps 600 \
  --episodes 4 \
  --seeds 801 802 803 804 805 806 807 808 809 810 \
  --no-render \
  --compare-base \
  --psb-report-label cpm_mixed_10seeds
```

P3.3 比 P3.2 多一个严格 `efficacy_gate`。最终通过必须先满足非劣与绝对车辆预算，
并至少满足以下一项：

1. `reward_lower_bound > 0`；
2. `vehicle.difference_upper_bound < 0` 且奖励非劣。

若 10-seed 结果接近零边界，则在不改变任何阈值的前提下扩展到 20 seeds；若仍未通过，
不应仅依靠旧 noninferiority 标志宣称 P3.3 有效。

## 7. P3 收口边界

P3.3 是 P3 的最终结构版本。P3.0 的配对接口、P3.1 的三通道 Critic、P3.2-C 的车辆
Primal-Dual 约束和本阶段的同步差分 PPO 已构成完整 P3 链路。后续不再增加新的 P3
网络模块或 `P3.3-X` 结构版本；只允许在冻结结构后集中调整学习率、更新次数、rollout
长度、两时间尺度和方差控制，并统一执行多训练种子、跨场景及消融实验。

结构修改后只做短冒烟验证，必须检查：训练/加载可完成、梯度有限、Base 冻结、联合
episode 边界一致、同步 reset 观测一致、近端根残差与扇区动作约束满足。冒烟结果不用于
宣称 Candidate 优于 Base，正式性能结论只来自最后锁定配置的系统实验。

```bash
python main_training.py \
  --config configs/psb_marl/p3_3_paired_differential_primal_dual_ppo.json \
  --iterations 1
```

该命令生成的 run 只属于结构冒烟；由于训练预算与锁定的 20-iteration 配置不同，验证器
不会把它当作正式 efficacy 候选。
