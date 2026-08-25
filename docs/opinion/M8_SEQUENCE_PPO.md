# M8：可微分 Sequence PPO

> 状态：实现完成；性能由用户手动训练验证  
> 前置阶段：M7 Sequence Buffer  
> 配置：`configs/opinion/m8_sequence_ppo.json`

## 1. M8 改变了什么

M7 只保证训练数据仍保留时间顺序，EvidenceNet 处于冻结状态。M8 第一次让 PPO 的
Actor loss 沿下面的完整链路反向传播：

```text
PPO advantage
  → 当前动作 log-prob
  → 有界速度 residual
  → z_t
  → 固定 OpinionDynamics 的多步展开
  → EvidenceNet 输出 b_t
  → EvidenceNet 参数
```

参数边界固定为：

| 模块 | M8 状态 | 梯度来源 |
|---|---|---|
| Base Actor | 冻结 | 无 |
| EvidenceNet | 训练 | Sequence PPO Actor loss + 证据正则项 |
| OpinionDynamics | 固定、无参数 | 只传递梯度 |
| OpinionResidual | 固定、无参数 | 只传递梯度 |
| Central Critic | 训练 | Critic loss |

Critic 仍只读取原始联合 observation。Critic loss 与 EvidenceNet 的计算图分离，不会
训练 EvidenceNet。

## 2. 序列重算

`OpinionSequenceBuffer.iter_sequence_minibatches()` 按真实 chunk 长度分桶并生成
`[chunk,time]` mini-batch。短尾不会作为 padding 进入损失。

每个 chunk 保存的 `z_init/edge_active_init` 都被 detach，它们是 truncated BPTT 的
边界。chunk 内：

1. Base Actor 与 EvidenceNet 在 chunk/time 两个维度上批量计算；
2. 只在短时间维上循环 OpinionDynamics；
3. 按 global neighbor ID 重放 reset、非候选衰减、gather、更新和 scatter；
4. 用最终 `loc/scale` 重建 TanhNormal，并对采样动作重算 log-prob；
5. 使用 clipped PPO objective 更新 EvidenceNet。

首个时间步的 `z_init` 已包含本步 reset 和非候选衰减，因此不会重复衰减。后续时间步
完整重放 M6 rollout 的状态语义。

## 3. 损失

```text
L = L_actor_PPO + L_entropy + L_critic
    + λ_neutral · mean((1-urgency) · b²)
    + λ_magnitude · mean(b²)
```

两个证据正则项只在有效 pair 上统计。它们抑制低紧迫度下的无意义强证据和全局幅值
膨胀，不使用人工 priority/leader/topology 标签。

## 4. 完整训练与测试

M8 默认从 `outputs/opinion/m7_sequence_buffer/` 中最近的 M7 run 初始化。正式实验应先
完成 M7；若 M7 尚在训练但已有匹配的中间 Policy/Critic，开发流程可继续并会输出警告。

```bash
conda activate sigmarl-nod

python main_training_opinion.py \
  --config configs/opinion/m8_sequence_ppo.json

python main_testing_opinion.py \
  --config configs/opinion/m8_sequence_ppo.json
```

精确测试某个 M8 run：

```bash
python main_testing_opinion.py \
  --config configs/opinion/m8_sequence_ppo.json \
  --run-dir outputs/opinion/m8_sequence_ppo/runs/<run-id>
```

开发用短配置为 `configs/opinion/m8_sequence_ppo_pilot.json`，它要求先存在对应的 M7
pilot 产物。

## 5. 训练产物与关键指标

每个 run 继续保存：

```text
final_policy.pth
final_critic.pth
final_evidence_net.pth
final_opinion_policy.pth
final_checkpoint.pt
final_opinion_state.pt
training_curves.pdf
metrics.json
config_source.json / config_resolved.json / opinion_config_resolved.json
```

M8 新增的主要诊断：

- `evidence_gradient_norm`：应为有限值，不能长期恒为 0；
- `sequence_approx_kl`、`sequence_clip_fraction`：观察 PPO 更新幅度；
- `sequence_log_prob_abs_error`：第一批更新前应接近 0；
- `sequence_state_replay_abs_error`：参数尚未更新时应接近 0，epoch 内随参数更新可上升；
- `evidence_neutral_penalty`、`evidence_magnitude_penalty`：证据正则项；
- `stateful_evidence_frozen=false`：确认 M8 已开放 EvidenceNet。

单次训练能证明流程闭环，不能证明性能优于 Base。性能结论仍需相同预算、多 seed 的
M9/M11 对比。

## 6. 实现时已完成的工程验证

2026-08-25 使用 `sigmarl-nod` 环境完成：

- Opinion 与入口测试共 40 项通过；
- M8 展开结果与 M6 tracker 在槽位交换、mask 变化和 agent reset 下逐步一致；
- chunk 最后一步的 `z` 可向 chunk 早期 `b` 反向传播；
- 真实 `CPM_mixed` 两个训练 iteration 连续完成，第二轮继续使用更新后的 EvidenceNet；
- 两轮 `evidence_gradient_norm` 分别约为 `0.0333` 与 `0.0225`，均为有限非零值；
- 最终 Policy/Critic/EvidenceNet/checkpoint/metrics/PDF 均可保存；
- `main_testing_opinion.py` 可加载 M8 final policy 并进入环境 rollout。无显示器的自动
  环境会在 pyglet 可视化初始化处停止，本地桌面环境不受这一 headless 限制。

以上只验证实现闭环，不构成性能提升结论。
