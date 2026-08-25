# M7：连续 Sequence Buffer

> 实现状态：代码、配置、指标和参考测试已完成；性能由用户手动验证  
> 策略边界：继承完整 M6 Policy/Critic，Base Actor 与 EvidenceNet 继续冻结  
> 下一阶段：M8 批量 chunk Sequence PPO

## 1. M7 改变什么

M6 已经在真实 rollout 中维护连续意见，但原 PPO 仍执行：

```text
[environment,time] → reshape(-1) → 随机单步 minibatch
```

M7 只在 `stage=sequence` 时替换为：

```text
[environment,time]
  → 按 done / trajectory 边界切分
  → 固定上限 chunk_length=16
  → chunk 之间随机打乱
  → chunk 内物理时间顺序保持不变
```

Base、M4、M5、M6 仍走各自原有路径，不构造 Sequence Buffer。

## 2. Chunk 数据合同

每个 chunk 保留普通 MAPPO 字段以及：

```text
pair_features, neighbor_ids, pair_mask,
urgency, confidence, agent_reset_mask,
z_prev, z_next, old_log_prob,
z_init, edge_active_init, valid_step_mask
```

`z_init [B_chunk,N,N]` 和 `edge_active_init [B_chunk,N,N]` 来自 chunk
第一个真实物理步之前的状态快照。两者同时保存，是因为仅有 `z_init` 无法区分一条边
是持续激活还是重新出现。

所有 rollout 张量在进入 Buffer 时 detach。M7 当前把短尾段保留为逻辑 padding，
`valid_step_mask=false` 的位置不物化进 Critic minibatch，因此不会进入 loss。

## 3. 边界合同

- 不跨并行 environment；
- 不跨 episode `done`；
- 不跨 `collector/traj_ids` 的变化；
- partial agent reset 由每步 `agent_reset_mask` 表达，不错误拼接旧车辆状态；
- 一个 rollout batch 的尾部可作为截断 chunk，下个 batch 用新的真实 `z_init` 开始。

## 4. M7 为什么仍不训练 EvidenceNet

M7 建立合法的数据容器，但尚未实现 M8 的可微时间展开。因此参数合同仍是：

```text
Base Actor       frozen
EvidenceNet      frozen
OpinionDynamics  fixed
OpinionResidual  fixed
Central Critic   trainable
```

Critic 不依赖意见历史，可以把每组完整 chunk 中的有效步临时合并后计算现有 value
loss。Actor loss 虽可被计算用于兼容 PPO 输出，但没有任何可训练 Actor 参数。

## 5. M6 来源合同

M7 配置中的：

```json
"sequence_ppo": {
    "enabled": true,
    "source_output_root": "outputs/opinion/m6_stateful_opinion/",
    "chunk_length": 16
}
```

必须指向兼容 M6 Stateful run。入口加载严格匹配的完整 M6 Policy 与 Critic，新的
rollout 从全零状态开始，不恢复 M6 结束时的交通状态。

## 6. 手动训练与测试

完整预算：

```bash
conda activate sigmarl-nod
cd /Users/zhangxiaotong/Code/sigmarl-nod

python main_training_opinion.py \
  --config configs/opinion/m7_sequence_buffer.json

python main_testing_opinion.py \
  --config configs/opinion/m7_sequence_buffer.json
```

Pilot 需要先存在对应的 Base、M5 和 M6 pilot 产物：

```bash
python main_training_opinion.py \
  --config configs/opinion/m7_sequence_buffer_pilot.json
```

## 7. 训练指标

每个 iteration 的 `metrics.json` 新增：

```text
sequence_chunk_count
sequence_valid_steps
sequence_padded_steps
sequence_valid_step_fraction
sequence_boundary_violation_count
sequence_state_memory_mb
```

核心要求：

```text
sequence_boundary_violation_count == 0
sequence_valid_steps == frames_per_batch
0 < sequence_valid_step_fraction <= 1
```

M7 不以超过 Base/M6 为目标，Reward 和碰撞率应大致保持 M6 的执行水平。

## 8. 参考检查

```bash
python -m unittest tests.opinion.test_m7_sequence_buffer
```

覆盖 environment/done/trajectory 边界、`z_init` 对齐、短尾 mask、有效步采样和旧
log-prob 有限性。M8 将在此基础上实现时间维展开、重算 log-prob 和 Evidence 梯度。

本次工程验证结果：

```text
完整 unittest：35 passed
真实 Collector 初始化：通过
真实 4096-step rollout：289 chunks / 4096 valid / 528 logical padding
boundary violations：0
临时目录 1 iteration × 1 epoch：GAE、Critic backward、optimizer、保存全部通过
```

该单轮 smoke 只验证训练链路，不作为性能结论。
