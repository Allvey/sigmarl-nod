# PSB-MARL P0：Base Passthrough

P0 只验证统一入口、配置、checkpoint 来源和 artifact 合同，不运行 PPO，也不改变
Base-MAPPO 的网络或动作。`final_policy.pth` 与选定 Base Actor 字节级相同，
`final_critic.pth` 与选定 Base Critic 字节级相同。

## 训练入口

```bash
conda activate sigmarl-nod
python main_training.py \
  --config configs/psb_marl/p0_base_passthrough.json
```

命令会打印唯一的 run 目录，并生成：

```text
config_source.json
config_resolved.json
psb_config_resolved.json
final_policy.pth
final_critic.pth
final_checkpoint.pt
p0_equivalence.json
deployment_manifest.json
training_status.json
validation_protocol.json
artifacts_manifest.json
```

必须检查 `p0_equivalence.json`：

```text
policy_bytes_identical = true
critic_bytes_identical = true
action_path = base_mappo_only
trainable_psb_parameters = 0
```

## 单场景可视化

```bash
python main_testing.py \
  --config configs/psb_marl/p0_base_passthrough.json \
  --run-dir <RUN_DIR> \
  --scenario intersection_2 \
  --max-steps 600 \
  --episodes 1 \
  --seeds 101 \
  --render \
  --compare-base
```

测试入口会先重新计算源 Base 与 P0 checkpoint 的 SHA-256，再运行实际道路环境。

## 无渲染多环境检查

```bash
python main_testing.py \
  --config configs/psb_marl/p0_base_passthrough.json \
  --run-dir <RUN_DIR> \
  --scenario CPM_mixed \
  --max-steps 600 \
  --episodes 4 \
  --seeds 101 102 103 104 105 \
  --no-render \
  --compare-base \
  --promote-if-noninferior
```

P0 不需要用有限样本估计性能差异。只要 policy checkpoint 字节相同，两者定义的
动作分布就完全相同；多环境 rollout 用于检查模型能够正常加载、奖励和动作均有限。

通过条件：

- 六项 checkpoint/hash 检查均为 `true`；
- `noninferiority_result` 为 `proven_by_identical_policy_checkpoint`；
- `nonfinite_action_count` 和 `nonfinite_reward_count` 均为零；
- 可视化行为与原 Base 一致。
