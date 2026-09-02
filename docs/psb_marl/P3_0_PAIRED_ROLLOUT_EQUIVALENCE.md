# PSB-MARL P3.0：配对 Rollout 等价性桥

P3.0 是 P2 与真正 P3 学习之间的只读桥梁。它不更新 Actor、Critic 或近端分岔层，
目标是先锁定一个可复现的 Candidate/Base 配对采样合同，避免后续差分 Critic 与
primal-dual PPO 在错误数据上训练。

## 1. 固定的来源

P3.0 配置为：

```text
configs/psb_marl/p3_0_paired_rollout_equivalence.json
```

它只接受同时满足下列条件的 P2 父运行：

1. P2 训练状态为完成，checkpoint hash 与 manifest 一致；
2. Actor 使用 P2.1-U 的 `supported_sector_q_gate`；
3. 动作修正为 `longitudinal_only`，且不修改动作方差；
4. Base fallback 与原始 Base checkpoint 完全一致；
5. 父运行被一个通过的 P2.2-R 三训练种子汇总唯一认证；
6. P3 配置复述的 P2 runtime 与父运行逐字段一致。

这些检查在加载配置时完成。更换父运行或任何 P2 超参数都必须同时提供与它匹配的
P2.2-R 认证结果。

## 2. P3.0 数据流

```text
同一 evaluation seed
        │
        ├── P3 packaged Candidate ── action/reward ──┐
        ├── original P2 Candidate ── action/reward ──┤ exact equality gate
        │                                             │
        └── Frozen Base ──────────── action/reward ───┘
                                  │
                                  ▼
               Candidate-minus-Base paired batch
```

配对 batch 暴露以下张量：

- Candidate 与 Base 的局部观测；
- Candidate 的 `z_next_dense` 分岔状态；
- Candidate 与 Base 的动作和奖励；
- Candidate 与 Base 的 `done` / `terminated` 边界；
- `delta_reward = reward_candidate - reward_base`；
- vehicle、lane 与 total collision 的逐步差分。

这正是 P3.1 差分 Critic 的输入/监督边界。P3.0 只验证张量形状和有限性，不训练网络，
也不把大体积 rollout 张量写入 checkpoint。

## 3. 安全语义

P3.0 运行目录同时保存：

- `candidate_policy.pth` / `candidate_critic.pth`：P2 候选的精确副本；
- `source_p2_policy.pth` / `source_p2_critic.pth`：显式来源副本；
- `base_fallback_policy.pth` / `base_fallback_critic.pth`：Base 精确副本；
- `final_policy.pth` / `final_critic.pth`：仍指向 Base fallback；
- `p3_0_equivalence.json`：checkpoint 字节等价证明。

因此 P3.0 不改变当前可部署策略，也不会把尚未经过 P3 非劣验证的 Candidate 自动部署。

## 4. 手动验证

先运行统一训练入口。该命令只打包文件，不进行环境训练：

```bash
python main_training.py \
  --config configs/psb_marl/p3_0_paired_rollout_equivalence.json
```

记下输出的 P3 run directory，然后执行快速配对测试：

```bash
python main_testing.py \
  --config configs/psb_marl/p3_0_paired_rollout_equivalence.json \
  --run-dir <P3_0_RUN_DIR> \
  --checkpoint <P3_0_RUN_DIR>/candidate_policy.pth \
  --scenario intersection_2 \
  --max-steps 128 \
  --episodes 2 \
  --seeds 601 602 \
  --no-render \
  --compare-base
```

通过标准是 `<P3_0_RUN_DIR>/p3_0_paired_equivalence.json` 中：

```json
{
  "passed": true,
  "source_equivalence_passed": true,
  "paired_contract_passed": true
}
```

其中每个 `source_equivalence` 条目还必须同时满足：

```text
shape_match = true
actions_exactly_equal = true
rewards_exactly_equal = true
max_abs_action_difference = 0
max_abs_reward_difference = 0
```

`paired_comparisons` 中 Candidate 与 Base 的差异不要求为零；它应与原 P2.1-U 相对
Base 的行为一致。P3.0 的“不弱于 baseline”保障来自 `final_policy.pth` 继续保留 Base，
而不是要求尚未优化的 P3 Candidate 在这一步产生新的性能提升。

## 5. 下一阶段

P3.0 通过后再进入 P3.1：冻结 Actor，只训练 Base-relative differential Critic，目标为
配对回报差 `G_candidate - G_base`。P3.1 的 Critic 校准通过后，P3.2 才打开受近端分岔
扇区约束的 Actor PPO 更新和 primal-dual 乘子更新。
