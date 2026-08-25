# M6：Stateful Collector 与连续意见状态

> 实现状态：代码、配置、参考测试和文档已完成；运行由用户手动执行  
> 训练边界：Base Actor 与 M5 EvidenceNet 冻结，仅训练原结构 Central Critic  
> 后续状态：M7 Sequence Buffer 已完成，下一阶段为 M8 Sequence PPO

## 1. 本阶段完成的因果链

M5 的 `z_direct=b` 被替换为：

```text
z_dense[E,N,N]
  → 按 neighbor_ids gather z_prev[E,N,K]
  → EvidenceNet 得到当前 b
  → Fixed OpinionDynamics(z_prev,b) 得到 z_next
  → OpinionResidual(z_next) 得到速度 residual
  → scatter z_next 回 z_dense
```

`utilities/opinion/state.py` 中的 `OpinionStateTracker` 是 rollout 状态所有者；
`StatefulOpinionPolicyBridge` 仍是纯函数，不在模块内部保存历史。PPO loss 只读取 rollout
保存的 `z_prev` 重算单步分布，不会更新 `z_dense`，因此一个物理步只积分一次。
Collector 初始化时为预分配输出而进行的策略探测调用会在完成后立即 `reset_state()`，
不计为物理步，也不会污染第一步真实 rollout。

M4 信息经过 `DiscreteDTypeCastTransform` 暴露为真正的离散 TensorSpec：
`neighbor_ids` 为 categorical `long`，`pair_mask/agent_reset_mask` 为 binary `bool`。
这避免 TorchRL Collector 探测策略时对整数 ID 调用连续分布 `randn`。

## 2. global-ID 与 reset 合同

状态表固定为：

```text
z_dense[e,i,j] = 环境 e 中车辆 i 对车辆 j 的有向意见
```

- 候选槽位交换时，通过 `neighbor_ids` 找回同一车辆状态；
- 新冲突边的 `z_prev` 为零；
- 失活边按固定 Dynamics 衰减，离开候选集合的状态也只衰减一次；
- `agent_reset_mask[e,i]` 同时清空第 `i` 行和第 `i` 列；
- episode done 清空对应环境切片；
- 对角线始终为零，不同并行环境完全隔离。

## 3. 为什么 M6 冻结 EvidenceNet

原 SigmaRL PPO 会把 rollout 展平成随机单步 minibatch。M6 虽然已经正确采集连续意见，
但还没有 M7 的连续 chunk 和 M8 的时间展开 loss。如果此时更新 EvidenceNet，PPO
重算将缺少完整前序计算图。

因此 M6 明确采用：

```text
Base Actor       冻结
M5 EvidenceNet   冻结
OpinionDynamics  固定、无训练参数
OpinionResidual  固定、无训练参数
Central Critic   训练（仍只读取原始联合 observation）
```

M6 用于验证状态因果、执行和可视化，不用于宣称 Evidence 已完成序列优化。

## 4. checkpoint 来源

M6 同时需要：

1. 与配置兼容的 Base run；
2. `stateful.evidence_output_root` 下的 M5 run；
3. M5 的独立 EvidenceNet 与同 reward/final Critic；
4. M5 run 中保留的、用于生成该 EvidenceNet 的同一个冻结 Base Actor。

允许的 M5 配对为：

```text
final_evidence_net.pth + final_critic.pth
reward<X>_evidence_net.pth + reward<X>_critic.pth
```

冻结 Base Actor 优先读取 M5 在启动时保存的 `source_base_actor.pth`。为兼容较早、
已经完整结束的 M5 运行，也接受其中内容等价的 `final_base_actor.pth`；因此无需仅为
文件名升级而重新训练已有 M5。旧快照中缺失或为 `null` 的 `stateful` 字段明确按
M5 的 `enabled=false` 读取。若两个 Base Actor 快照都不存在，才要求重新启动 M5。

旧 M5 run 如果只有完整 `reward<X>_policy.pth`、没有独立 EvidenceNet，需要用当前代码
重新启动一次 M5；入口会明确拒绝含糊提取。

## 5. 用户手动训练

pilot 顺序：

```bash
conda activate sigmarl-nod

python main_training.py --config configs/base/pilot.json
python main_training_opinion.py \
  --config configs/opinion/m5_direct_evidence_pilot.json
python main_training_opinion.py \
  --config configs/opinion/m6_stateful_opinion_pilot.json
```

如果前一阶段已经有兼容 checkpoint，可以跳过对应训练。完整预算 M6：

```bash
python main_training_opinion.py \
  --config configs/opinion/m6_stateful_opinion.json
```

## 6. 用户手动可视化测试

```bash
python main_testing_opinion.py \
  --config configs/opinion/m6_stateful_opinion.json
```

测试指定 run 或中间策略：

```bash
python main_testing_opinion.py \
  --config configs/opinion/m6_stateful_opinion.json \
  --run-dir outputs/opinion/m6_stateful_opinion/runs/<run-id>

python main_testing_opinion.py \
  --config configs/opinion/m6_stateful_opinion.json \
  --checkpoint outputs/opinion/m6_stateful_opinion/runs/<run-id>/reward<X>_policy.pth
```

每次测试从全零 `z_dense` 开始，不继承训练结束时的交通状态。

## 7. 可视化与指标

车辆面板新增：

```text
z_prev, z_next, delta_z, q, normalized weight
```

训练指标新增：

```text
stateful_z_abs_mean
stateful_z_max_abs
stateful_delta_z_abs_mean
agent_reset_fraction
stateful_evidence_frozen=true
```

最终 run 额外保存：

```text
final_opinion_state.pt   # 仅用于诊断的 terminal z_dense/edge_active
final_checkpoint.pt      # 同时包含 terminal_opinion_state
```

这不是部署时必须恢复的模型权重；独立测试默认从零意见状态开始。Exact resume 留给 M9。

## 8. 参考测试与人工验收

参考测试：

```bash
python -m unittest tests.opinion.test_m6_stateful_opinion
```

重点检查：

- 候选槽位交换后意见仍跟随 global ID；
- partial reset 清空相关行列；
- 冲突重新出现时从零开始；
- pure bridge 重复输入得到相同输出；
- Base Actor/EvidenceNet 均保持冻结；
- 长 rollout 中 `z` 有限、无跨环境串扰；
- 可视化只读取诊断张量，不再次更新状态。

按照项目约定，本 Session 不执行训练、测试或性能判断。

## 9. TorchRL 0.2.1 初始化兼容

若旧实现启动 Collector 时出现：

```text
RuntimeError: "normal_kernel_cpu" not implemented for 'Long'
```

原因不是模型或权重，而是 TorchRL 把只修改了 dtype 的连续 observation spec 当作
整数连续分布，并在策略探测阶段执行 `torch.randn(dtype=torch.long)`。当前实现通过两点
消除了该路径：

1. `StatefulOpinionPolicyController` 明确提供 `in_keys/out_keys`，作为 TensorDict policy
   直接交给 Collector；
2. `neighbor_ids` 与两个 mask 使用真正的离散 TensorSpec，而不是连续 TensorSpec
   仅改变 dtype。

另外，TorchRL 可能把每车一个 reset 标志表示为 `[E,N,1]`，状态边界会将这个明确的
标量尾维规范化为 `[E,N]`；其他不匹配形状仍会被拒绝。

修复不改变邻车选择、意见方程、动作输出或优化器。
