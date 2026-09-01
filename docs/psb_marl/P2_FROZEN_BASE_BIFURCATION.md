# PSB-MARL P2：Frozen-Base Bifurcation PPO

P2 首次开放可学习的分岔控制和分支条件动作分布，但 Base Actor 全程冻结。实现结构为：

```text
局部物理观测 o_i ───────────────> Frozen Base Actor ──> (mu_base, sigma_base) ─┐
                                                                            │
局部冲突对 (chi_ij, z_ij) ─> shared PairScorer                              │
                  └─────────> swap PairScorer ─> antisymmetric b_ij          │
                                                   │                         │
                                                   v                         │
                         Proximal Saturating Bifurcation ─> z_next ─> q_ij   │
                                                                          v v
                                     masked local edge aggregation ─> branch adapter
                                                                          │
                                                                          v
                                                               (mu, sigma) -> action

训练专用：Central Critic(x, stop_gradient(z)) -> augmented return / GAE
```

这里没有动作残差，也没有固定的“正意见加速、负意见减速”规则。意见分支只形成可学习
的上下文，最终由通用分布适配器同时调整动作均值和尺度。PairScorer、边聚合器和适配器
均共享参数；执行时不使用 Critic，也不引入显式 agent-to-agent 消息。

## 1. P2 的训练约束

- Base Actor 从已验证的 P1/Base checkpoint 加载并冻结；
- `PairScorer(chi,z)-PairScorer(swap(chi),-z)` 构造交换反对称、有界的 `b`；
- 近端层在前向求收敛根，反向使用隐式 Jacobian；
- rollout 保存完整边状态，PPO 更新按连续 chunk 重算 `b,z`，chunk 内不 detach；
- GAE 和 Critic 使用增广奖励 `task_reward-lambda_b*mean(b^2)`；
- 同一能量代价还通过显式路径梯度更新控制网络；
- Critic 读取 `stop_gradient(z)`，价值回归不能修改分岔状态；
- 独立的 `b` 信赖域和饱和惩罚限制单次策略更新。

分支适配器最后一层严格零初始化。因此训练第 0 步满足：

```text
candidate action distribution == Base action distribution
```

即使初始 `b,z` 非零，在适配器开始学习前也不会改变物理动作。

## 2. 安全部署约束

P2 训练结束后同时保存两套策略：

```text
candidate_policy.pth       # 学到的完整 P2 Actor，仅作为候选
candidate_critic.pth       # 增广 Critic
base_fallback_policy.pth   # 与 Base 字节完全相同
base_fallback_critic.pth   # 与 Base 字节完全相同
final_policy.pth           # 默认仍指向 Base fallback
final_critic.pth           # 默认仍指向 Base fallback
```

因此“每个阶段至少不弱于 Base”由部署门保证，而不是在训练完成时假定候选一定更好。
只有候选通过手动成对非劣检验并显式执行 promotion 后，`final_policy.pth` 才切换为
候选；任何拒绝都会恢复 Base fallback。

## 3. 训练

正式训练：

```bash
conda activate sigmarl-nod

python main_training.py \
  --config configs/psb_marl/p2_frozen_base_bifurcation.json
```

若想先做一次短流程检查：

```bash
python main_training.py \
  --config configs/psb_marl/p2_frozen_base_bifurcation.json \
  --iterations 1
```

短流程成功后，可在同一 run 上恢复到配置的 250 次迭代：

```bash
python main_training.py \
  --config configs/psb_marl/p2_frozen_base_bifurcation.json \
  --resume <RUN_DIR>/latest_checkpoint.pt \
  --iterations 250
```

训练时主要检查 `metrics.json`：

```text
base_actor_frozen = true
rollout_z_antisymmetry_error <= 1e-6
rollout_max_root_residual <= 1e-6
rollout_min_root_denominator > 0
sequence_state_replay_abs_error 接近 0
sequence_log_prob_abs_error 在每轮首个 epoch 接近 0
control_gradient_norm 和 adapter_gradient_norm 为有限值
```

`sequence_log_prob_abs_error` 在后续 PPO epoch 会随着参数更新增大，这是正常的旧策略—新
策略差异；同时观察 `sequence_approx_kl`、`sequence_clip_fraction` 和
`control_trust`，判断更新是否过猛。

P2.1-D 已补充活跃边、临界区、分支承诺和 `q=0` 反事实旁路诊断。指标定义与验证方式
见 [`P2_1_DIAGNOSTICS.md`](P2_1_DIAGNOSTICS.md)。这些指标只读，不改变本阶段训练合同。

P2.1-C 在保留本阶段作为通用-adapter消融组的同时，新增严格满足
`q=0 => candidate distribution == Base distribution` 的因果门控策略。结构、配置隔离
和验证命令见 [`P2_1_CAUSAL_BRANCH.md`](P2_1_CAUSAL_BRANCH.md)。

P2.1-S 进一步冻结 Base 动作尺度，只允许因果分支修正动作均值；尺度输出头在结构上
不存在，promotion 也会检查尺度逐元素严格等于 Base。结构和验证命令见
[`P2_1_MEAN_ONLY.md`](P2_1_MEAN_ONLY.md)。

## 4. 候选完整性测试

终端打印 `<RUN_DIR>` 后，先只测试候选，不做 promotion：

```bash
python main_testing.py \
  --config configs/psb_marl/p2_frozen_base_bifurcation.json \
  --run-dir <RUN_DIR> \
  --checkpoint <RUN_DIR>/candidate_policy.pth \
  --scenario CPM_mixed \
  --max-steps 128 \
  --episodes 1 \
  --seeds 101 \
  --no-render
```

检查 `p2_manual_validation.json` 中：动作、奖励、`z` 均无非有限值，`b` 未越界，根残差
不超过配置容差，根分母为正，反对称误差不超过 `1e-6`。

## 5. 与 Base 的正式比较

配置要求至少 5 个 paired seeds。建议每个 seed 使用 4 个并行 episode：

```bash
python main_testing.py \
  --config configs/psb_marl/p2_frozen_base_bifurcation.json \
  --run-dir <RUN_DIR> \
  --scenario CPM_mixed \
  --max-steps 600 \
  --episodes 4 \
  --seeds 101 102 103 104 105 \
  --no-render \
  --compare-base
```

每个 seed 下候选和 Base 在 rollout 前重新使用相同随机种子。报告计算：

```text
reward lower bound = mean(candidate - Base) - z_conf * standard_error
collision upper bound = mean(candidate - Base) + z_conf * standard_error
```

只有同时满足：

```text
reward lower bound >= -reward_margin
collision upper bound <= collision_margin
全部结构/数值检查通过
```

才得到 `passed_paired_confidence_bounds`。当前容差为每 agent-step 奖励 `0.002`、总碰撞率
`0.002`；它们是有限样本非劣界，不应解释为候选已经显著优于 Base。

## 6. 显式 promotion

确认上述报告合理后，重复同一正式协议并加入：

```bash
--promote-if-noninferior
```

通过时：

```text
promotion_result = candidate_promoted
deployment_manifest.selected = candidate_promoted
final_policy.pth 与 candidate_policy.pth 字节相同
```

未通过或 seed 数不足时，命令会以非零状态结束，并保持：

```text
promotion_result = rejected_base_fallback_retained
final_policy.pth 与 base_fallback_policy.pth 字节相同
```

最后可在不 promotion 的情况下补充 `intersection_2` 等跨场景比较。跨场景结果应单独
报告，不与训练场景的 paired confidence bound 混为一个统计量。
